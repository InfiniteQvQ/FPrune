import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

# ✅ 加载 LLaMA 7B 模型（从本地路径）
cache_dir = "/root/autodl-tmp/llm_weights"
model = AutoModelForCausalLM.from_pretrained("pinkmanlove/llama-7b-hf", 
                                             cache_dir=cache_dir, device_map="auto", torch_dtype=torch.float16)

def get_layer_index(name):
    """
    从模块或参数名称中提取层索引
    例如 "model.model.layers.0.self_attn.q_proj" 提取出 "0"
    """
    parts = name.split(".")
    if "layers" in parts:
        idx = parts.index("layers")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return None

# ================================
# **步骤 1: 计算 Q/K 层的 Hessian (ESD)**
# ================================
def compute_qk_esd(model):
    """ 计算每层 Q/K 层的 Hessian (ESD) 重要性，返回格式：{layer_index: {"q_proj": value, "k_proj": value}} """
    esd_values = {}

    for name, module in model.named_modules():
        if "q_proj" in name or "k_proj" in name:
            layer_index = get_layer_index(name)
            if layer_index is None:
                continue
            # 初始化该层的字典
            if layer_index not in esd_values:
                esd_values[layer_index] = {}
            # 根据名称确定是 q_proj 还是 k_proj
            if "q_proj" in name:
                proj_type = "q_proj"
            elif "k_proj" in name:
                proj_type = "k_proj"
            else:
                continue

            # 复制权重并转为 float32
            matrix = module.weight.data.clone().cpu().to(torch.float32)
            # 计算奇异值平方
            eigs = torch.square(torch.linalg.svdvals(matrix).flatten())
            spectral_norm = eigs[-1].item()
            log_nz_eigs = torch.log(eigs)
            # 计算 alpha（注意：这里的公式可能需要根据实际需求调整）
            alpha = 1 + len(eigs) / (torch.sum(log_nz_eigs) - len(eigs) * log_nz_eigs[0])
            esd_value = alpha.item() * torch.log10(torch.tensor(spectral_norm)).item()
            esd_values[layer_index][proj_type] = esd_value

    print("ESD 计算完成: ", esd_values)
    return esd_values

# ================================
# **步骤 2: 计算 V, Output, Gate, Up, Down 层的 GradNorm**
# ================================
def compute_vgoud_gradnorm(model):
    """ 计算每层 V, Output, Gate, Up, Down 层的 GradNorm 重要性，返回格式：
        {layer_index: {"v_proj": value, "o_proj": value, "gate_proj": value, "up_proj": value, "down_proj": value}} 
    """
    gradnorm_values = {}

    for name, param in model.named_parameters():
        if any(proj in name for proj in ["v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]):
            layer_index = get_layer_index(name)
            if layer_index is None:
                continue
            if layer_index not in gradnorm_values:
                gradnorm_values[layer_index] = {}
            # 确定投影类型
            if "v_proj" in name:
                proj_type = "v_proj"
            elif "o_proj" in name:
                proj_type = "o_proj"
            elif "gate_proj" in name:
                proj_type = "gate_proj"
            elif "up_proj" in name:
                proj_type = "up_proj"
            elif "down_proj" in name:
                proj_type = "down_proj"
            else:
                continue

            gradnorm_val = torch.norm(param.data, p=2).item()
            gradnorm_values[layer_index][proj_type] = gradnorm_val

    print("GradNorm 计算完成: ", gradnorm_values)
    return gradnorm_values

# ================================
# **步骤 3: 计算动态剪枝权重**
# ================================
def compute_dynamic_weights(esd_values, gradnorm_values):
    """
    计算全局动态剪枝权重
    lambda_qk: 基于所有层 Q/K 的 ESD 之和
    lambda_vgoud: 基于所有层 V/Gate/Up/Down 的 GradNorm 之和
    """
    esd_list = [val for layer in esd_values.values() for val in layer.values()]
    gradnorm_list = [val for layer in gradnorm_values.values() for val in layer.values()]

    esd_sum = sum(esd_list) if esd_list else 1e-8
    gradnorm_sum = sum(gradnorm_list) if gradnorm_list else 1e-8

    lambda_qk = esd_sum / (esd_sum + gradnorm_sum)
    lambda_vgoud = gradnorm_sum / (esd_sum + gradnorm_sum)

    print(f"Q/K 剪枝权重: {lambda_qk}, V/Gate/Up/Down 剪枝权重: {lambda_vgoud}")
    return lambda_qk, lambda_vgoud

# ================================
# **步骤 4: 计算剪枝评分**
# ================================
def compute_layer_pruning_score(model):
    """ 计算 Transformer 结构中每层的剪枝评分，评分 = 动态权重 * exp(对应层的重要性) """
    esd_values = compute_qk_esd(model)
    gradnorm_values = compute_vgoud_gradnorm(model)
    lambda_qk, lambda_vgoud = compute_dynamic_weights(esd_values, gradnorm_values)

    pruning_scores = {}

    # 针对 Q/K 层
    for layer, proj_dict in esd_values.items():
        if layer not in pruning_scores:
            pruning_scores[layer] = {}
        for proj, val in proj_dict.items():
            # 使用 torch.exp 计算指数变换
            score = lambda_qk * torch.exp(torch.tensor(val, dtype=torch.float32))
            pruning_scores[layer][proj] = score.item()

    # 针对 V, Output, Gate, Up, Down 层
    for layer, proj_dict in gradnorm_values.items():
        if layer not in pruning_scores:
            pruning_scores[layer] = {}
        for proj, val in proj_dict.items():
            score = lambda_vgoud * torch.exp(torch.tensor(val, dtype=torch.float32))
            pruning_scores[layer][proj] = score.item()

    return pruning_scores

# ================================
# **步骤 6: 运行剪枝**
# ================================
# 计算剪枝评分
pruning_scores = compute_layer_pruning_score(model)
print("各层剪枝评分: ", pruning_scores)
