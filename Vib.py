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
    torch_dtype=torch.float16
)

# 计算特征值谱熵
def spectral_entropy(matrix):
    """ 计算特征值谱熵 (Spectral Entropy) """
    matrix = matrix.detach().cpu().numpy()
    eigenvalues = np.abs(np.linalg.eigvals(matrix))
    eigenvalues = eigenvalues / np.sum(eigenvalues)  # 归一化
    return entropy(eigenvalues)  # 计算熵

# 计算 MLP 层的奇异值谱熵
def svd_entropy(matrix):
    """ 计算奇异值谱熵 (Singular Value Entropy) """
    matrix = matrix.detach().cpu().numpy()
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    singular_values = singular_values / np.sum(singular_values)  # 归一化
    return entropy(singular_values)  # 计算熵

# 存储层重要性
layer_importance = {}

for layer_idx, layer in enumerate(model.model.layers):
    # Attention 层
    q_proj = layer.self_attn.q_proj.weight
    k_proj = layer.self_attn.k_proj.weight
    v_proj = layer.self_attn.v_proj.weight
    attn_entropy = (spectral_entropy(q_proj) + spectral_entropy(k_proj) + spectral_entropy(v_proj)) / 3

    # MLP 层
    fc1 = layer.mlp.fc1.weight
    fc2 = layer.mlp.fc2.weight
    mlp_entropy = (svd_entropy(fc1) + svd_entropy(fc2)) / 2

    # 参数归一化 (避免 MLP 层参数较多导致偏差)
    num_params_attn = q_proj.numel() + k_proj.numel() + v_proj.numel()
    num_params_mlp = fc1.numel() + fc2.numel()
    total_params = num_params_attn + num_params_mlp

    # 计算归一化层重要性
    attn_weight = num_params_attn / total_params
    mlp_weight = num_params_mlp / total_params
    layer_score = attn_weight * attn_entropy + mlp_weight * mlp_entropy
    layer_importance[layer_idx] = layer_score

    print(layer_score)

print("final: ")

print(layer_score)