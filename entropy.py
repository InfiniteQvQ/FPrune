import torch
import numpy as np
from transformers import AutoModelForCausalLM

# 🎯 计算 Spectral Entropy（衡量信息分布）
def compute_spectral_entropy(weight_matrix):
    """计算谱熵"""
    eigs = torch.linalg.svdvals(weight_matrix).pow(2)  # 计算特征值平方
    eigs = eigs / eigs.sum()  # 归一化
    entropy = -torch.sum(eigs * torch.log(eigs + 1e-9))  # 避免 log(0)
    return entropy.item()

# 🛠️ 加载 LLaMA 2 (7B)
print("\n🚀 Loading LLaMA 2 (7B)...")
model_2_7b = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    cache_dir="/root/autodl-tmp/llm_weights",
    device_map="auto",
    torch_dtype=torch.float16
)

# 🎯 计算 LLaMA 2 (7B) 的 V Entropy
print("\n📊 Calculating LLaMA 2 (7B) V Entropy...")
v_entropy_2_7b = []
for idx, layer in enumerate(model_2_7b.model.layers):
    v_weight = layer.self_attn.v_proj.weight.float().cpu()
    entropy = compute_spectral_entropy(v_weight)
    v_entropy_2_7b.append(entropy)
    print(f"Layer {idx} V Entropy: {entropy:.4f}")

# 🔥 清空 CUDA Cache
del model_2_7b
torch.cuda.empty_cache()
print("\n🧹 CUDA Cache Cleared!\n")

# 🛠️ 加载 LLaMA 7B
print("\n🚀 Loading LLaMA 7B...")
model_7b = AutoModelForCausalLM.from_pretrained(
    "pinkmanlove/llama-7b-hf",
    cache_dir="/root/autodl-tmp/llm_weights",
    device_map="auto",
    torch_dtype=torch.float16
)

# 🎯 计算 LLaMA 7B 的 V Entropy
print("\n📊 Calculating LLaMA 7B V Entropy...")
v_entropy_7b = []
for idx, layer in enumerate(model_7b.model.layers):
    v_weight = layer.self_attn.v_proj.weight.float().cpu()
    entropy = compute_spectral_entropy(v_weight)
    v_entropy_7b.append(entropy)
    print(f"Layer {idx} V Entropy: {entropy:.4f}")

# 🔥 清空 CUDA Cache
del model_7b
torch.cuda.empty_cache()
print("\n🧹 CUDA Cache Cleared!\n")

# 🚀 结果对比
print("\n🔍 **LLaMA 2 (7B) vs LLaMA 7B V Entropy Comparison**")
print(f"LLaMA 2 (7B) V Entropy (Avg): {np.mean(v_entropy_2_7b):.4f}")
print(f"LLaMA 7B V Entropy (Avg): {np.mean(v_entropy_7b):.4f}")
