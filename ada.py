import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

# ✅ 加载 LLaMA 7B 模型（从本地路径）
cache_dir = "/root/autodl-tmp/llm_weights"
model = AutoModelForCausalLM.from_pretrained("pinkmanlove/llama-7b-hf", 
                                             cache_dir=cache_dir, device_map="auto", torch_dtype=torch.float16)

# ================================
# **步骤 1: 计算 Q/K 层的 Hessian (ESD)**
# ================================
def compute_qk_esd(model):
    """ 计算每层 Q/K 层的 Hessian (ESD) 重要性 """
    esd_values = {}

    for name, module in model.named_modules():
        if "q_proj" in name or "k_proj" in name:  # 只计算 Q/K 层
            parts = name.split(".")  # 分割 `layers.0.attention.q_proj`
            if len(parts) < 3:
                continue  # 跳过异常层

            layer_index = parts[1]  # 取 `layers.0` 中的 `0`
            key = f"layer_{layer_index}_{parts[-1]}"  # 例如 "layer_0_q_proj"

            matrix = module.weight.data.clone().cpu().to(torch.float32)
            eigs = torch.square(torch.linalg.svdvals(matrix).flatten())
            spectral_norm = eigs[-1].item()

            log_nz_eigs = torch.log(eigs)
            alpha = 1 + len(eigs) / (torch.sum(log_nz_eigs) - len(eigs) * log_nz_eigs[0])
            esd_values[key] = alpha.item() * torch.log10(torch.tensor(spectral_norm))

    print("ESD 计算完成: ", esd_values)
    return esd_values



# ================================
# **步骤 2: 计算 V, Output, Gate, Up, Down 层的 GradNorm**
# ================================
def compute_vgoud_gradnorm(model):
    """ 计算每层 V, Output, Gate, Up, Down 层的 GradNorm 重要性 """
    gradnorm_values = {}

    for name, param in model.named_parameters():
        if any(layer in name for layer in ["v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]):
            parts = name.split(".")  # 分割 `layers.0.attention.v_proj`
            if len(parts) < 3:
                continue  # 跳过异常层

            layer_index = parts[1]  # 取 `layers.0` 中的 `0`
            key = f"layer_{layer_index}_{parts[-1]}"  # 例如 "layer_0_v_proj"

            gradnorm_values[key] = torch.norm(param.data, p=2).item()

    print("GradNorm 计算完成: ", gradnorm_values)
    return gradnorm_values



# ================================
# **步骤 3: 计算动态剪枝权重**
# ================================
def compute_dynamic_weights(model):
    """ 计算每层的动态剪枝权重 """
    esd_values = compute_qk_esd(model)
    gradnorm_values = compute_vgoud_gradnorm(model)

    esd_sum = sum(esd_values.values()) if esd_values else 1e-8
    gradnorm_sum = sum(gradnorm_values.values()) if gradnorm_values else 1e-8

    lambda_qk = esd_sum / (esd_sum + gradnorm_sum)
    lambda_vgoud = gradnorm_sum / (esd_sum + gradnorm_sum)

    print(f"Q/K 剪枝权重: {lambda_qk}, V/Gate/Up/Down 剪枝权重: {lambda_vgoud}")
    return lambda_qk, lambda_vgoud


# ================================
# **步骤 4: 计算剪枝评分**
# ================================
def compute_layer_pruning_score(model):
    """ 计算 Transformer 结构中的剪枝评分 """
    lambda_qk, lambda_vgoud = compute_dynamic_weights(model)
    esd_values = compute_qk_esd(model)
    gradnorm_values = compute_vgoud_gradnorm(model)

    pruning_scores = {}

    for name in esd_values:
        pruning_scores[name] = lambda_qk * torch.exp(torch.tensor(esd_values[name], dtype=torch.float32))

    for name in gradnorm_values:
        pruning_scores[name] = lambda_vgoud * torch.exp(torch.tensor(gradnorm_values[name], dtype=torch.float32))

    return pruning_scores



# ================================
# **步骤 6: 运行剪枝**
# ================================
# 计算剪枝评分
pruning_scores = compute_layer_pruning_score(model)

print(pruning_scores)
