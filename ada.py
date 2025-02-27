import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

# -----------------------------
# 1. 加载 LLaMA 7B 模型（从本地路径）
# -----------------------------
cache_dir = "/root/autodl-tmp/llm_weights"
model = AutoModelForCausalLM.from_pretrained(
    "pinkmanlove/llama-7b-hf",
    cache_dir=cache_dir,
    device_map="auto",
    torch_dtype=torch.float16
)

# -----------------------------
# 辅助函数：从模块名称中提取层索引
# 例如："model.model.layers.0.self_attn.q_proj" 提取出 "0"
# -----------------------------
def get_layer_index(name):
    parts = name.split(".")
    if "layers" in parts:
        idx = parts.index("layers")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return None

# -----------------------------
# 2. 计算 Q/K 层的 ESD
# -----------------------------
def compute_qk_esd(model, eps=1e-8):
    """
    遍历模型中所有 q_proj 或 k_proj 模块，计算其 ESD 值。
    返回格式： { layer_index: {"q_proj": esd_value, "k_proj": esd_value} }
    注意：如果权重张量维度小于2（例如偏置），则跳过计算。
    """
    esd_values = {}
    for name, module in model.named_modules():
        
            layer_index = get_layer_index(name)
            if layer_index is None:
                continue
            if layer_index not in esd_values:
                esd_values[layer_index] = {}
            proj_type = "q_proj" if "q_proj" in name else "k_proj"
            matrix = module.weight.data.clone().cpu().to(torch.float32)
            if matrix.ndim < 2:
                continue  # 跳过非二维张量
            # 计算奇异值，再取平方
            eigs = torch.square(torch.linalg.svdvals(matrix).flatten())
            spectral_norm = eigs[-1].item()  # 取最后一个奇异值（平方后）
            log_nz_eigs = torch.log(eigs + eps)
            # 计算一个 alpha 参数，公式可调整
            alpha = 1 + len(eigs) / (torch.sum(log_nz_eigs) - len(eigs) * log_nz_eigs[0])
            esd = alpha.item() * torch.log10(torch.tensor(spectral_norm) + eps).item()
            esd_values[layer_index][proj_type] = esd
    print("ESD computed for Q/K:", esd_values)
    return esd_values

# -----------------------------
# 3. 仅基于 Q/K 计算每层的重要性
# -----------------------------
def compute_qk_importance(model, qk_multiplier=1.0):
    """
    利用 compute_qk_esd 计算每层 q_proj 与 k_proj 的 ESD，
    对于每一层，将 q_proj 与 k_proj 的 ESD 取平均，
    并取负（因为 ESD 数值越低表示重要性越高），再乘以 qk_multiplier。
    返回格式： { layer_index: importance }
    """
    esd_values = compute_qk_esd(model)
    importance = {}
    for layer, proj_dict in esd_values.items():
        values = []
        if "q_proj" in proj_dict:
            values.append(proj_dict["q_proj"])
        if "k_proj" in proj_dict:
            values.append(proj_dict["k_proj"])
        if values:
            avg_esd = sum(values) / len(values)
            importance[layer] = -avg_esd * qk_multiplier
    return importance

# -----------------------------
# 4. 主函数：计算并按层输出重要性
# -----------------------------
def main():
    # 这里 qk_multiplier 可以调节 Q/K 权重（默认设为 1.0，可根据需要调整）
    for name, module in model.model.layers[0].named_modules():
        if any(x in name for x in ["q_proj", "k_proj", "v_proj", "gate_proj", "o_proj", "up_proj", "down_proj"]):
            print(name)
    importance = compute_qk_importance(model, qk_multiplier=1.0)
    

    # 假设层索引为数字字符串，按数值从小到大排序输出
    #sorted_importance = [importance[layer] for layer in sorted(importance, key=lambda x: int(x))]
    #print("Importance per layer (based solely on Q/K):")
    #print(sorted_importance)

if __name__ == "__main__":
    main()
