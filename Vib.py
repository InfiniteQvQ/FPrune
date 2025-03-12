import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
from transformers import AutoModelForCausalLM

# 加载 LLaMA-7B
cache_dir = "/root/autodl-tmp/llm_weights"
model = AutoModelForCausalLM.from_pretrained(
    "pinkmanlove/llama-7b-hf",
    cache_dir=cache_dir,
    device_map="auto",  # 让 Hugging Face 自动分配多个 GPU
    torch_dtype=torch.float32
)

def singular_value_spectrum(weight_matrix):
    """计算 SVD 奇异值谱"""
    weight_matrix = weight_matrix.float()  # 避免 float16 报错
    U, S, V = torch.linalg.svd(weight_matrix.cpu().detach(), full_matrices=False)
    return S.numpy()

def esd_spectrum(weight_matrix):
    """计算特征值谱分布 (ESD)"""
    weight_matrix = weight_matrix.float()  # 避免 float16 报错
    eigvals = np.abs(np.linalg.eigvals((weight_matrix @ weight_matrix.T).cpu().detach().numpy()))
    return eigvals

layer_importance_scores = {}

for layer_idx, layer in enumerate(model.model.layers):
    print(f"Processing Layer {layer_idx}...")

    # 🧠 Attention 层
    q_proj = layer.self_attn.q_proj.weight
    k_proj = layer.self_attn.k_proj.weight
    v_proj = layer.self_attn.v_proj.weight
    attn_score = np.mean([
        np.sum(singular_value_spectrum(q_proj)), 
        np.sum(singular_value_spectrum(k_proj)), 
        np.sum(singular_value_spectrum(v_proj))
    ])  # SVD 计算重要性

    # 🔥 MLP 层
    gate_proj = layer.mlp.gate_proj.weight
    up_proj = layer.mlp.up_proj.weight
    down_proj = layer.mlp.down_proj.weight
    mlp_score = np.mean([
        np.sum(esd_spectrum(gate_proj)), 
        np.sum(esd_spectrum(up_proj)), 
        np.sum(esd_spectrum(down_proj))
    ])  # ESD 计算重要性

    # 🎯 Output 层
    output_proj = layer.self_attn.o_proj.weight
    output_score = np.sum(singular_value_spectrum(output_proj))  # SVD 计算重要性

    # 📊 计算相对重要性
    layer_relative_importance = attn_score + mlp_score + output_score
    layer_importance_scores[layer_idx] = layer_relative_importance

# 🚀 排序
sorted_layers = sorted(layer_importance_scores.items(), key=lambda x: x[1], reverse=True)

print("\n🔝 LLaMA 7B 每层的相对重要性排序:")
for idx, importance in sorted_layers:
    print(f"Layer {idx}: {importance:.4f}")