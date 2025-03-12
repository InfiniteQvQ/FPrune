import torch
import numpy as np
from transformers import AutoModelForCausalLM

# 🛠️ 加载 LLaMA-7B
model = AutoModelForCausalLM.from_pretrained(
    "pinkmanlove/llama-7b-hf",
    cache_dir="/root/autodl-tmp/llm_weights",
    device_map="auto",
    torch_dtype=torch.float16
)

# 🎯 计算 PL_Alpha_Hill
def pl_alpha_hill(weight_matrix, k_ratio=0.1):
    """计算 Hill 估计的 PL_Alpha_Hill"""
    weight_matrix = weight_matrix.float()
    with torch.no_grad():
        eigvals = torch.linalg.eigvalsh(weight_matrix @ weight_matrix.T).cpu().numpy()
    eigvals = np.sort(eigvals)[::-1]  # 降序排列
    n = len(eigvals)
    k = int(k_ratio * n)  # 取前 k% 计算
    if k < 1: return 1.0
    lambda_n_k = eigvals[k-1]
    log_ratio = np.log(eigvals[:k]) - np.log(lambda_n_k)
    alpha_hill = 1 + k / np.sum(log_ratio)
    return alpha_hill

# 🎯 计算单层重要性
def process_layer(layer_idx, layer):
    print(f"Processing Layer {layer_idx}...")

    # 🧠 Q, K, V (Attention)
    attn_hill = np.mean([
        pl_alpha_hill(layer.self_attn.q_proj.weight),
        pl_alpha_hill(layer.self_attn.k_proj.weight),
        pl_alpha_hill(layer.self_attn.v_proj.weight)
    ])

    # 🔥 MLP 层（Gate, Up, Down）
    mlp_hill = np.mean([
        pl_alpha_hill(layer.mlp.gate_proj.weight),
        pl_alpha_hill(layer.mlp.up_proj.weight),
        pl_alpha_hill(layer.mlp.down_proj.weight)
    ])

    # 🎯 Output 层
    output_hill = pl_alpha_hill(layer.self_attn.o_proj.weight)

    # 📊 计算相对重要性
    layer_relative_importance = attn_hill + mlp_hill + output_hill
    print(layer_relative_importance)
    return layer_idx, layer_relative_importance

# 🚀 计算所有层的重要性
layer_importance_scores = []
for idx, layer in enumerate(model.model.layers):
    layer_importance_scores.append(process_layer(idx, layer))

# 🚀 归一化
scores = torch.tensor([imp[1] for imp in layer_importance_scores])
s1, s2 = 0.8, 1.2
max_score, min_score = scores.max(), scores.min()
normalized_scores = ((scores - min_score) / (max_score - min_score)) * (s2 - s1) + s1

normalized_scores = 0.72 * normalized_scores
print(normalized_scores)
print(normalized_scores.mean())
print("\n🔝 LLaMA 7B 每层的归一化相对重要性:")
for (idx, _), importance in zip(layer_importance_scores, normalized_scores.tolist()):
    print(f"Layer {idx}: {importance:.4f}")
