import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM

# 🎯 计算 Spectral Entropy（衡量信息密度）
def compute_spectral_entropy(weight_matrix):
    """计算谱熵（信息分布的均匀性）"""
    eigs = torch.linalg.svdvals(weight_matrix).pow(2)
    eigs = eigs / eigs.sum()
    entropy = -torch.sum(eigs * torch.log(eigs + 1e-9))
    return entropy.item()

# 🎯 计算 Rank（衡量权重矩阵的独立性）
def compute_rank(weight_matrix, threshold=1e-5):
    """计算权重矩阵的 Rank（非零奇异值个数）"""
    singular_values = torch.linalg.svdvals(weight_matrix)
    rank = (singular_values > threshold).sum().item()
    return rank / weight_matrix.shape[1]  # 归一化 Rank

# 🎯 计算 QK Entropy（衡量 Q/K 影响）
def compute_qk_entropy(layer):
    """计算 QK 矩阵的谱熵"""
    q_weight = layer.self_attn.q_proj.weight.cpu()
    k_weight = layer.self_attn.k_proj.weight.cpu()
    qk_matrix = q_weight @ k_weight.T  # 计算 QK 相关性
    return compute_spectral_entropy(qk_matrix)

# 🚀 处理单个模型
def process_model(model_name, model_path):
    print(f"\n🚀 Loading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        cache_dir="/root/autodl-tmp/llm_weights",
        device_map="auto",
        torch_dtype=torch.float16
    )

    # 🎯 计算所有层的 V Entropy、V Rank、QK Entropy
    v_entropy, v_rank, qk_entropy = [], [], []
    
    for idx, layer in enumerate(model.model.layers):
        print(f"Processing Layer {idx}...")
        
        v_weight = layer.self_attn.v_proj.weight.float().cpu()
        v_entropy.append(compute_spectral_entropy(v_weight))
        v_rank.append(compute_rank(v_weight))

        qk_entropy.append(compute_qk_entropy(layer))

    # 🔥 释放显存
    del model
    torch.cuda.empty_cache()
    print("\n🧹 CUDA Cache Cleared!\n")

    return np.array(v_entropy), np.array(v_rank), np.array(qk_entropy)

# 🚀 计算 LLaMA 2 (7B) 和 LLaMA 7B
v_entropy_2_7b, v_rank_2_7b, qk_entropy_2_7b = process_model("LLaMA 2 (7B)", "meta-llama/Llama-2-7b-hf")
v_entropy_7b, v_rank_7b, qk_entropy_7b = process_model("LLaMA 7B", "pinkmanlove/llama-7b-hf")

# 🚀 画图
layers = np.arange(len(v_entropy_2_7b))
plt.figure(figsize=(10, 6))
plt.plot(layers, v_entropy_2_7b, label="LLaMA 2 (7B) V Entropy", linestyle="--", marker="o")
plt.plot(layers, v_entropy_7b, label="LLaMA 7B V Entropy", linestyle="--", marker="s")
plt.xlabel("Layer")
plt.ylabel("V Entropy")
plt.title("V Entropy Comparison")
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(layers, v_rank_2_7b, label="LLaMA 2 (7B) V Rank", linestyle="--", marker="o")
plt.plot(layers, v_rank_7b, label="LLaMA 7B V Rank", linestyle="--", marker="s")
plt.xlabel("Layer")
plt.ylabel("V Rank")
plt.title("V Rank Comparison")
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(layers, qk_entropy_2_7b, label="LLaMA 2 (7B) QK Entropy", linestyle="--", marker="o")
plt.plot(layers, qk_entropy_7b, label="LLaMA 7B QK Entropy", linestyle="--", marker="s")
plt.xlabel("Layer")
plt.ylabel("QK Entropy")
plt.title("QK Entropy Comparison")
plt.legend()
plt.show()

# 🚀 结果对比
print("\n🔍 **LLaMA 2 (7B) vs LLaMA 7B Comparison**")
print(f"LLaMA 2 (7B) V Entropy (Avg): {np.mean(v_entropy_2_7b):.4f}")
print(f"LLaMA 7B V Entropy (Avg): {np.mean(v_entropy_7b):.4f}")
print(f"LLaMA 2 (7B) V Rank (Avg): {np.mean(v_rank_2_7b):.4f}")
print(f"LLaMA 7B V Rank (Avg): {np.mean(v_rank_7b):.4f}")
print(f"LLaMA 2 (7B) QK Entropy (Avg): {np.mean(qk_entropy_2_7b):.4f}")
print(f"LLaMA 7B QK Entropy (Avg): {np.mean(qk_entropy_7b):.4f}")
