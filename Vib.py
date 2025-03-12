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
    return np.sum(S.cpu().numpy())  # 返回 SVD 奇异值总和

# 🎯 计算 ESD（完整特征值谱）
def esd_spectrum(weight_matrix):
    """计算完整 ESD（特征值谱分布）"""
    weight_matrix = weight_matrix.float()
    with torch.no_grad():
        eigvals = torch.linalg.eigvalsh(weight_matrix @ weight_matrix.T)
    return np.max(eigvals.cpu().numpy())  # 只取最大特征值

# 🎯 计算单层重要性
def process_layer(layer_idx, layer):
    print(f"Processing Layer {layer_idx}...")

    # 🧠 Attention 层（SVD 衡量）
    q_proj = layer.self_attn.q_proj.weight
    k_proj = layer.self_attn.k_proj.weight
    v_proj = layer.self_attn.v_proj.weight
    attn_svd = np.mean([
        singular_value_spectrum(q_proj),
        singular_value_spectrum(k_proj),
        singular_value_spectrum(v_proj)
    ])  # ✅ SVD 衡量信息传播能力

    # 🔥 MLP 层（ESD 反向衡量）
    gate_proj = layer.mlp.gate_proj.weight
    up_proj = layer.mlp.up_proj.weight
    down_proj = layer.mlp.down_proj.weight
    mlp_esd = np.mean([
        esd_spectrum(gate_proj),
        esd_spectrum(up_proj),
        esd_spectrum(down_proj)
    ])  # ✅ 计算最大特征值（代表可能的冗余性）

    # 🎯 Output 层（SVD 衡量）
    output_proj = layer.self_attn.o_proj.weight
    output_svd = singular_value_spectrum(output_proj)  # ✅ SVD 衡量

    # 📊 计算相对重要性
    layer_relative_importance = attn_svd * 0.1 + (1 / mlp_esd) * 0.9 + output_svd * 0.1  # 归一化权重计算

    # 🚀 释放显存
    del q_proj, k_proj, v_proj, gate_proj, up_proj, down_proj, output_proj
    torch.cuda.empty_cache()

    return layer_idx, layer_relative_importance

# 🚀 计算所有层的重要性
layer_importance_scores = []
for idx, layer in enumerate(model.model.layers):
    layer_importance_scores.append(process_layer(idx, layer))


print("\n🔝 LLaMA 7B 每层的相对重要性排序:")
for idx, importance in layer_importance_scores:
    print(f"Layer {idx}: {importance:.4f}")
