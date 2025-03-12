import torch
import numpy as np
from transformers import AutoModelForCausalLM

# 🛠️ 设置缓存目录
cache_dir = "/root/autodl-tmp/llm_weights"

# 🚀 加载 LLaMA-7B（自动分配 GPU）
model = AutoModelForCausalLM.from_pretrained(
    "pinkmanlove/llama-7b-hf",
    cache_dir=cache_dir,
    device_map="auto",
    torch_dtype=torch.float16  # ✅ 用 float16 降低显存占用
)

# 🎯 计算 SVD（完整奇异值谱）
def singular_value_spectrum(weight_matrix):
    """计算完整 SVD 奇异值谱"""
    weight_matrix = weight_matrix.float()
    with torch.no_grad():
        U, S, V = torch.linalg.svd(weight_matrix, full_matrices=False)
    return S.cpu().numpy()  # 返回 SVD 奇异值

# 🎯 计算谱熵（Spectral Entropy）
def spectral_entropy(singular_values):
    """计算谱熵"""
    normalized_sv = singular_values / singular_values.sum()
    return -np.sum(normalized_sv * np.log(normalized_sv + 1e-9))

# 🎯 计算 ESD（完整特征值谱）
def esd_spectrum(weight_matrix):
    """计算完整 ESD（特征值谱分布）"""
    weight_matrix = weight_matrix.float()
    with torch.no_grad():
        eigvals = torch.linalg.eigvalsh(weight_matrix @ weight_matrix.T)
    return eigvals.cpu().numpy()  # 返回完整特征值

# 🎯 计算单层重要性
def process_layer(layer_idx, layer):
    print(f"Processing Layer {layer_idx}...")

    # 🧠 Attention 层（SVD + 谱熵）
    q_proj = layer.self_attn.q_proj.weight
    k_proj = layer.self_attn.k_proj.weight
    v_proj = layer.self_attn.v_proj.weight
    attn_svd_entropy = np.mean([
        spectral_entropy(singular_value_spectrum(q_proj)),
        spectral_entropy(singular_value_spectrum(k_proj)),
        spectral_entropy(singular_value_spectrum(v_proj))
    ])  # ✅ SVD + 谱熵 计算信息传播能力

    # 🔥 MLP 层（归一化 ESD）
    gate_proj = layer.mlp.gate_proj.weight
    up_proj = layer.mlp.up_proj.weight
    down_proj = layer.mlp.down_proj.weight
    mlp_esd = np.mean([
        np.max(esd_spectrum(gate_proj)),
        np.max(esd_spectrum(up_proj)),
        np.max(esd_spectrum(down_proj))
    ])  # ✅ 取最大特征值

    # 🎯 Output 层（SVD + 谱熵）
    output_proj = layer.self_attn.o_proj.weight
    output_svd_entropy = spectral_entropy(singular_value_spectrum(output_proj))  # ✅ SVD + 谱熵 计算

    # 📊 计算相对重要性
    layer_relative_importance = attn_svd_entropy * 0.5 - (mlp_esd * 0.5) + output_svd_entropy * 0.1

    

    # 🚀 释放显存
    del q_proj, k_proj, v_proj, gate_proj, up_proj, down_proj, output_proj
    torch.cuda.empty_cache()

    return layer_idx, layer_relative_importance

# 🚀 计算所有层的重要性
layer_importance_scores = []
for idx, layer in enumerate(model.model.layers):
    layer_importance_scores.append(process_layer(idx, layer))

# 🚀 归一化（0.8 ~ 1.2 范围）
scores = torch.tensor([imp[1] for imp in layer_importance_scores])
s1, s2 = 0.8, 1.2
max_score, min_score = scores.max(), scores.min()
normalized_scores = ((scores - min_score) / (max_score - min_score)) * (s2 - s1) + s1

# 🚀 排序
sorted_layers = sorted(zip([imp[0] for imp in layer_importance_scores], normalized_scores.tolist()), key=lambda x: x[1], reverse=True)

print("\n🔝 LLaMA 7B 每层的归一化相对重要性排序:")
for idx, importance in sorted_layers:
    print(f"Layer {idx}: {importance:.4f}")
