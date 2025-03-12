import torch
import numpy as np
import joblib  # ✅ 并行计算
from transformers import AutoModelForCausalLM

# 🛠️ 设置缓存目录
cache_dir = "/root/autodl-tmp/llm_weights"

# 🚀 加载 LLaMA-7B（自动分配 GPU）
model = AutoModelForCausalLM.from_pretrained(
    "pinkmanlove/llama-7b-hf",
    cache_dir=cache_dir,
    device_map="auto",   # ✅ 让 Hugging Face 自动管理 GPU
    torch_dtype=torch.float16  # ✅ 用 float16 降低显存占用
)

# 🎯 计算 SVD（完整奇异值谱）
def singular_value_spectrum(weight_matrix):
    """计算完整 SVD 奇异值谱"""
    weight_matrix = weight_matrix.float()
    with torch.no_grad():
        U, S, V = torch.linalg.svd(weight_matrix, full_matrices=False)  # ✅ 计算完整 SVD
    return S.cpu().numpy()  # ✅ 只在 CPU 上转换 numpy

# 🎯 计算 ESD（完整特征值谱）
def esd_spectrum(weight_matrix):
    """计算完整 ESD（特征值谱分布）"""
    weight_matrix = weight_matrix.float()
    with torch.no_grad():
        eigvals = torch.linalg.eigvalsh(weight_matrix @ weight_matrix.T)  # ✅ 计算完整特征值
    return eigvals.cpu().numpy()  # ✅ 只在 CPU 上转换 numpy

# 🎯 计算单层重要性（支持 GPU 计算）
def process_layer(layer_idx, layer):
    print(f"Processing Layer {layer_idx}...")

    # 🧠 Attention 层（完整 SVD）
    q_proj = layer.self_attn.q_proj.weight
    k_proj = layer.self_attn.k_proj.weight
    v_proj = layer.self_attn.v_proj.weight
    attn_score = np.mean([
        np.sum(singular_value_spectrum(q_proj)),
        np.sum(singular_value_spectrum(k_proj)),
        np.sum(singular_value_spectrum(v_proj))
    ])  # ✅ SVD 计算重要性

    # 🔥 MLP 层（完整 ESD）
    gate_proj = layer.mlp.gate_proj.weight
    up_proj = layer.mlp.up_proj.weight
    down_proj = layer.mlp.down_proj.weight
    mlp_score = np.mean([
        np.sum(esd_spectrum(gate_proj)),
        np.sum(esd_spectrum(up_proj)),
        np.sum(esd_spectrum(down_proj))
    ])  # ✅ ESD 计算重要性

    # 🎯 Output 层（完整 SVD）
    output_proj = layer.self_attn.o_proj.weight
    output_score = np.sum(singular_value_spectrum(output_proj))  # ✅ SVD 计算重要性

    # 📊 计算相对重要性
    layer_relative_importance = attn_score + mlp_score + output_score

    # 🚀 释放 GPU 显存
    del q_proj, k_proj, v_proj, gate_proj, up_proj, down_proj, output_proj
    torch.cuda.empty_cache()

    return layer_idx, layer_relative_importance  # ✅ 返回结果

# 🚀 逐层计算（**GPU 支持**）
layer_importance_scores = []
for idx, layer in enumerate(model.model.layers):
    layer_importance_scores.append(process_layer(idx, layer))

# 🚀 排序
sorted_layers = sorted(layer_importance_scores, key=lambda x: x[1], reverse=True)

print("\n🔝 LLaMA 7B 每层的相对重要性排序:")
for idx, importance in sorted_layers:
    print(f"Layer {idx}: {importance:.4f}")
