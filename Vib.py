import torch
import numpy as np
from transformers import AutoModelForCausalLM

# ✅ 自动选择 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ 加载 LLaMA-7B
cache_dir = "/root/autodl-tmp/llm_weights"
model = AutoModelForCausalLM.from_pretrained(
    "pinkmanlove/llama-7b-hf",
    cache_dir=cache_dir,
    device_map="auto",
    torch_dtype=torch.bfloat16
).to(device)  # 确保整个模型加载到 GPU

def singular_value_spectrum(weight_matrix):
    """计算 SVD 奇异值谱（在 GPU 运行）"""
    weight_matrix = weight_matrix.to(device).float()  # 确保在 GPU 上
    U, S, V = torch.linalg.svd(weight_matrix, full_matrices=False)  # 直接在 GPU 计算 SVD
    return S.detach().cpu().numpy()  # 计算完成后转换为 NumPy 数组

def esd_spectrum(weight_matrix):
    """计算特征值谱分布 (ESD)（在 GPU 运行）"""
    weight_matrix = weight_matrix.to(device).float()  # 确保在 GPU 上
    gram_matrix = weight_matrix @ weight_matrix.T  # 计算 Gram 矩阵
    eigenvalues, _ = torch.linalg.eigh(gram_matrix)  # GPU 计算特征值
    return eigenvalues.abs().detach().cpu().numpy()  # 计算完成后转换为 NumPy 数组

layer_importance_scores = {}

for layer_idx, layer in enumerate(model.model.layers):
    print(f"Processing Layer {layer_idx}...")

    # 🧠 Attention 层
    q_proj = layer.self_attn.q_proj.weight.to(device)
    k_proj = layer.self_attn.k_proj.weight.to(device)
    v_proj = layer.self_attn.v_proj.weight.to(device)
    attn_score = np.mean([
        np.sum(singular_value_spectrum(q_proj)), 
        np.sum(singular_value_spectrum(k_proj)), 
        np.sum(singular_value_spectrum(v_proj))
    ])  # SVD 计算重要性

    # 🔥 MLP 层
    gate_proj = layer.mlp.gate_proj.weight.to(device)
    up_proj = layer.mlp.up_proj.weight.to(device)
    down_proj = layer.mlp.down_proj.weight.to(device)
    mlp_score = np.mean([
        np.sum(esd_spectrum(gate_proj)), 
        np.sum(esd_spectrum(up_proj)), 
        np.sum(esd_spectrum(down_proj))
    ])  # ESD 计算重要性

    # 🎯 Output 层
    output_proj = layer.self_attn.o_proj.weight.to(device)
    output_score = np.sum(singular_value_spectrum(output_proj))  # SVD 计算重要性

    # 📊 计算相对重要性
    layer_relative_importance = attn_score + mlp_score + output_score
    layer_importance_scores[layer_idx] = layer_relative_importance

# 🚀 排序
sorted_layers = sorted(layer_importance_scores.items(), key=lambda x: x[1], reverse=True)

print("\n🔝 LLaMA 7B 每层的相对重要性排序:")
for idx, importance in sorted_layers:
    print(f"Layer {idx}: {importance:.4f}")
