import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
import numpy as np

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
# 2. 计算每层所有模块的 ESD（聚合后作为层级指标）
# -----------------------------
def compute_layer_esd(model, eps=1e-8):
    """
    遍历模型所有具有 weight 属性的模块，
    对每个模块计算 ESD（基于 SVD），
    然后将同一层内所有模块的 ESD 取平均，得到该层的整体 ESD 指标。
    返回格式： { layer_index: avg_esd_value }
    """
    layer_esd = {}
    for name, module in model.named_modules():
        if hasattr(module, "weight") and isinstance(module.weight, torch.Tensor):
            layer_index = get_layer_index(name)
            if layer_index is None:
                continue
            matrix = module.weight.data.clone().cpu().to(torch.float32)
            eigs = torch.square(torch.linalg.svdvals(matrix).flatten())
            spectral_norm = eigs[-1].item()  # 这里取最后一个奇异值（平方后）
            log_nz_eigs = torch.log(eigs + eps)
            # 计算一个简单的 alpha 参数（公式可根据需要调整）
            alpha = 1 + len(eigs) / (torch.sum(log_nz_eigs) - len(eigs) * log_nz_eigs[0])
            esd = alpha.item() * torch.log10(torch.tensor(spectral_norm) + eps).item()
            if layer_index not in layer_esd:
                layer_esd[layer_index] = []
            layer_esd[layer_index].append(esd)
    # 对每一层的 ESD 列表取平均
    averaged_esd = {}
    for layer, esd_list in layer_esd.items():
        averaged_esd[layer] = sum(esd_list) / len(esd_list)
    return averaged_esd

# -----------------------------
# 3. 计算每层符合条件模块的 GradNorm（取对数后平均）
# -----------------------------
def compute_layer_gradnorm(model, eps=1e-8):
    """
    遍历模型中所有参数名称中包含 ["v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    使用 get_layer_index 得到所属层，
    对每个参数计算 L2 范数并取对数，
    然后对同一层内的对数值取平均，得到该层的 GradNorm 指标。
    返回格式： { layer_index: avg_log_gradnorm }
    """
    layer_gradnorm = {}
    for name, param in model.named_parameters():
        if any(proj in name for proj in ["v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]):
            layer_index = get_layer_index(name)
            if layer_index is None:
                continue
            gradnorm_val = torch.norm(param.data, p=2).item()
            gradnorm_log = torch.log(torch.tensor(gradnorm_val + eps)).item()
            if layer_index not in layer_gradnorm:
                layer_gradnorm[layer_index] = []
            layer_gradnorm[layer_index].append(gradnorm_log)
    averaged_gradnorm = {}
    for layer, grad_list in layer_gradnorm.items():
        averaged_gradnorm[layer] = sum(grad_list) / len(grad_list)
    return averaged_gradnorm

# -----------------------------
# 4. 动态结合 ESD 与 GradNorm 计算每层的重要性
# -----------------------------
def compute_combined_layer_importance(model, eps=1e-8, qk_multiplier=1.5):
    """
    计算每层的综合重要性指标：
      ① 计算每层的 ESD，并取负后（因为原始 ESD 数值越低越好）乘以 qk_multiplier（增强 Q/K 影响）；
      ② 计算每层的 GradNorm，取对数后对所有层归一化到 [0,1]；
      ③ 动态计算全局权重 lambda_esd 和 lambda_grad，
          其中 lambda_esd = total_esd / (total_esd + total_grad)，lambda_grad 类似；
      ④ 最终每层重要性 = lambda_esd * importance_esd + lambda_grad * importance_grad。
    返回格式： { layer_index: combined_importance }
    """
    layer_esd = compute_layer_esd(model, eps)
    layer_grad = compute_layer_gradnorm(model, eps)
    
    # 对 ESD 部分取负（原始 ESD 越低表示越重要）并乘以 qk_multiplier
    importance_esd = {layer: -val * qk_multiplier for layer, val in layer_esd.items()}
    
    # 对 GradNorm 部分归一化到 [0,1]
    grad_values = list(layer_grad.values())
    min_grad = min(grad_values)
    max_grad = max(grad_values)
    importance_grad = {}
    for layer, val in layer_grad.items():
        importance_grad[layer] = (val - min_grad) / (max_grad - min_grad + eps)
    
    # 收集所有层
    all_layers = set(importance_esd.keys()) | set(importance_grad.keys())
    total_esd = sum(importance_esd.get(layer, 0) for layer in all_layers)
    total_grad = sum(importance_grad.get(layer, 0) for layer in all_layers)
    lambda_esd = total_esd / (total_esd + total_grad + eps)
    lambda_grad = total_grad / (total_esd + total_grad + eps)
    print("动态权重: lambda_esd =", lambda_esd, ", lambda_grad =", lambda_grad)
    
    combined_importance = {}
    for layer in all_layers:
        combined_importance[layer] = lambda_esd * importance_esd.get(layer, 0) + lambda_grad * importance_grad.get(layer, 0)
    return combined_importance

# -----------------------------
# 5. 主函数：运行全部计算并按层顺序输出结果
# -----------------------------
def main():
    combined_importance = compute_combined_layer_importance(model, qk_multiplier=1.5)
    # 假设层索引为字符串形式的数字，按 int 排序输出
    final_importance_list = [combined_importance[layer] for layer in sorted(combined_importance, key=lambda x: int(x))]
    print("各层最终重要性评分（按层排序）:")
    print(final_importance_list)

if __name__ == "__main__":
    main()
