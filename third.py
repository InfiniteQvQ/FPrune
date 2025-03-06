import torch
import math
import numpy as np
from transformers import AutoModelForCausalLM, LlamaTokenizer
import json

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "pinkmanlove/llama-7b-hf"  # 示例模型名称

print("Loading model...")
cache_dir = "/root/autodl-tmp/llm_weights"
model = AutoModelForCausalLM.from_pretrained(
    "pinkmanlove/llama-7b-hf",
    cache_dir=cache_dir,
    device_map="auto",  # 让 Hugging Face 自动分配多个 GPU
    torch_dtype=torch.float16
)
model.eval()
print("Model loaded.")

def compute_alpha(weight):
    """
    计算 Linear 层权重对应的 α 值：
      - 对权重进行 SVD 得到奇异值 s；
      - 将 s^2 作为特征值，选取大于阈值的特征值；
      - 对每个候选 xmin 计算 alpha = 1 + n / sum(log(eigs/xmin))，返回中值作为近似。
    """
    # 计算奇异值
    u, s, v = torch.linalg.svd(weight, full_matrices=False)
    eigs = s**2  # 特征值
    # 仅保留大于1e-6的值
    nz_eigs = eigs[eigs > 1e-6]
    if len(nz_eigs) == 0:
        return None
    nz_eigs, _ = torch.sort(nz_eigs)
    alphas = []
    for i in range(len(nz_eigs) - 1):
        xmin = nz_eigs[i]
        n = float(len(nz_eigs) - i)
        denom = torch.sum(torch.log(nz_eigs[i:] / xmin))
        if denom == 0:
            alpha = float('inf')
        else:
            alpha = 1 + n / denom
        alphas.append(alpha)
    if len(alphas) > 0:
        # 返回中位数作为近似
        median_alpha = alphas[len(alphas) // 2]
        return median_alpha
    else:
        return None

# 计算每个 Linear 层扰动前的 α 值
original_alphas = {}
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        weight = module.weight.data.float()
        alpha_val = compute_alpha(weight)
        if alpha_val is not None:
            original_alphas[name] = alpha_val
            print(f"Layer: {name}, original alpha: {alpha_val}")

epsilon = 1e-6
print("Perturbing model parameters with epsilon =", epsilon)
with torch.no_grad():
    for param in model.parameters():
        param.add_(epsilon * torch.randn_like(param))

# 计算扰动后的 α 值
perturbed_alphas = {}
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        weight = module.weight.data.float()
        alpha_val = compute_alpha(weight)
        if alpha_val is not None:
            perturbed_alphas[name] = alpha_val
            print(f"Layer: {name}, perturbed alpha: {alpha_val}")

# 计算三阶信息近似
third_order_esd = {}
for name in original_alphas:
    if name in perturbed_alphas:
        third_order = (perturbed_alphas[name] - original_alphas[name]) / epsilon
        # 如果 third_order 是张量，转为 Python 数值
        third_order_esd[name] = third_order.item() if isinstance(third_order, torch.Tensor) else third_order

print("\n========= Third-order ESD Approximation =========")
for layer, value in third_order_esd.items():
    print(f"{layer}: {value}")

# 保存结果到文件
with open("third_order_esd_results.json", "w") as f:
    json.dump(third_order_esd, f, indent=4)

print("Third-order ESD information saved successfully.")
