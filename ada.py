import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

# -----------------------------
# 1. 加载 LLaMA 7B 模型（从本地路径）
# -----------------------------
cache_dir = "/root/autodl-tmp/llm_weights"
model = AutoModelForCausalLM.from_pretrained("pinkmanlove/llama-7b-hf",
                                             cache_dir=cache_dir,
                                             device_map="auto",
                                             torch_dtype=torch.float16)

# -----------------------------
# 辅助函数：从名称中提取层索引
# 例如： "model.model.layers.0.self_attn.q_proj" 提取出 "0"
# -----------------------------
def get_layer_index(name):
    parts = name.split(".")
    if "layers" in parts:
        idx = parts.index("layers")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return None

# -----------------------------
# 2. 计算 Q/K 层的 ESD（Hessian 重要性）
# -----------------------------
def compute_qk_esd(model):
    """
    计算每层 Q/K 层的 ESD，返回字典格式：
      { layer_index: {"q_proj": esd_value, "k_proj": esd_value} }
    说明：
      - 由于对小于1的数取 log10 后结果为负，因此这里用 -ESD 表示重要性，
        数值越大代表越重要。
    """
    esd_values = {}
    for name, module in model.named_modules():
        if "q_proj" in name or "k_proj" in name:
            layer_index = get_layer_index(name)
            if layer_index is None:
                continue
            if layer_index not in esd_values:
                esd_values[layer_index] = {}
            proj_type = "q_proj" if "q_proj" in name else "k_proj"
            # 复制权重并转为 float32
            matrix = module.weight.data.clone().cpu().to(torch.float32)
            # 计算奇异值平方
            eigs = torch.square(torch.linalg.svdvals(matrix).flatten())
            spectral_norm = eigs[-1].item()  # 取最后一个奇异值的平方
            # 计算 alpha（公式可根据需要调整）
            log_nz_eigs = torch.log(eigs)
            alpha = 1 + len(eigs) / (torch.sum(log_nz_eigs) - len(eigs) * log_nz_eigs[0])
            esd = alpha.item() * torch.log10(torch.tensor(spectral_norm)).item()
            esd_values[layer_index][proj_type] = esd
    print("ESD 计算完成: ", esd_values)
    return esd_values

# -----------------------------
# 3. 计算 V、Output、Gate、Up、Down 层的 GradNorm
# -----------------------------
def compute_vgoud_gradnorm(model):
    """
    计算每层 V, Output, Gate, Up, Down 层的 GradNorm 重要性，返回字典格式：
      { layer_index: {"v_proj": val, "o_proj": val, "gate_proj": val, "up_proj": val, "down_proj": val} }
    """
    gradnorm_values = {}
    for name, param in model.named_parameters():
        if any(proj in name for proj in ["v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]):
            layer_index = get_layer_index(name)
            if layer_index is None:
                continue
            if layer_index not in gradnorm_values:
                gradnorm_values[layer_index] = {}
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
            gradnorm_values[layer_index][proj_type] = torch.norm(param.data, p=2).item()
    print("GradNorm 计算完成: ", gradnorm_values)
    return gradnorm_values

# -----------------------------
# 4. 动态计算最终每层的重要性评分
# -----------------------------
def compute_dynamic_importance(esd_values, gradnorm_values, eps=1e-8, qk_multiplier=1.5):
    """
    利用动态方法计算每层的重要性评分。
    处理步骤：
      a. 对 Q/K 部分：将 ESD 取相反数 (-ESD)，并对 q_proj 与 k_proj 取平均，再乘以 qk_multiplier（增大Q/K权重），得到 importance_qk。
      b. 对 GradNorm 部分：对每个 GradNorm 取对数后平均，得到 importance_grad，然后在所有层内归一化到 [0,1]。
      c. 动态权重：先计算所有层的 importance_qk 与 importance_grad 的全局和，
         动态计算 lambda_qk = total_qk / (total_qk + total_grad)， lambda_grad 同理。
      d. 最终每层重要性：final_importance = lambda_qk * importance_qk_layer + lambda_grad * importance_grad_layer
    返回：
      - final_importance: { layer_index: final_importance_value }
    """
    final_importance = {}
    importance_qk_dict = {}   # 每层 Q/K 重要性（数值越大表示越重要）
    importance_grad_dict = {} # 每层 GradNorm 重要性（经过对数和归一化后）

    layers = set(list(esd_values.keys()) + list(gradnorm_values.keys()))
    # 先计算 Q/K 重要性
    for layer in layers:
        if layer in esd_values:
            esd_layer = esd_values[layer]
            if "q_proj" in esd_layer and "k_proj" in esd_layer:
                imp_qk = - (esd_layer["q_proj"] + esd_layer["k_proj"]) / 2.0
            elif "q_proj" in esd_layer:
                imp_qk = - esd_layer["q_proj"]
            elif "k_proj" in esd_layer:
                imp_qk = - esd_layer["k_proj"]
            else:
                imp_qk = 0.0
        else:
            imp_qk = 0.0
        # 增加 Q/K 权重
        importance_qk_dict[layer] = qk_multiplier * imp_qk

    # 计算 GradNorm 重要性：先取对数再求平均
    temp_grad = {}
    for layer in layers:
        if layer in gradnorm_values:
            grad_vals = gradnorm_values[layer].values()
            grad_logs = [torch.log(torch.tensor(val, dtype=torch.float32) + eps).item() for val in grad_vals]
            temp_grad[layer] = sum(grad_logs) / len(grad_logs)
        else:
            temp_grad[layer] = 0.0

    # 对所有层的 gradnorm 值归一化到 [0,1]
    grad_list = list(temp_grad.values())
    min_grad = min(grad_list)
    max_grad = max(grad_list)
    for layer in layers:
        # 归一化公式
        importance_grad_dict[layer] = (temp_grad[layer] - min_grad) / (max_grad - min_grad + eps)

    # 动态计算全局权重
    total_qk = sum(importance_qk_dict.values())
    total_grad = sum(importance_grad_dict.values())
    lambda_qk = total_qk / (total_qk + total_grad + eps)
    lambda_grad = total_grad / (total_qk + total_grad + eps)
    print("动态权重: lambda_qk =", lambda_qk, ", lambda_grad =", lambda_grad)

    # 计算最终每层的重要性评分
    for layer in layers:
        final_importance[layer] = lambda_qk * importance_qk_dict[layer] + lambda_grad * importance_grad_dict[layer]
    return final_importance

# -----------------------------
# 5. 主函数：运行全部计算
# -----------------------------
def main():
    esd_values = compute_qk_esd(model)
    gradnorm_values = compute_vgoud_gradnorm(model)
    final_importance = compute_dynamic_importance(esd_values, gradnorm_values, qk_multiplier=1.5)
    # 按层号顺序输出（假设层索引是数字形式的字符串）
    final_importance_list = [final_importance[layer] for layer in sorted(final_importance, key=lambda x: int(x))]
    print("各层最终重要性评分（按层排序）:")
    print(final_importance_list)

if __name__ == "__main__":
    main()
