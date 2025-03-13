import torch
import numpy as np
from transformers import AutoModelForCausalLM
import math
# 🛠️ 加载 LLaMA-7B
model = AutoModelForCausalLM.from_pretrained(
    "pinkmanlove/llama-7b-hf",
    cache_dir="/root/autodl-tmp/llm_weights",
    device_map="auto",
    torch_dtype=torch.float16
)
metric = np.load("/root/autodl-tmp/alpha_peak.npy")

# 🎯 计算 ESD（最大特征值）
def esd_spectrum(weight_matrix):
    """计算最大特征值 (ESD)"""
    weight_matrix = weight_matrix.float()
    with torch.no_grad():
        eigvals = torch.linalg.eigvalsh(weight_matrix @ weight_matrix.T)
    return eigvals.max().cpu().numpy()  # 返回最大特征值

# 🎯 计算单层重要性
def process_layer(layer_idx, layer):
    print(f"Processing Layer {layer_idx}...")

    # 🧠 计算 Q, K, V, O 层的 Alpha-Hill 之和
    attn_hill_sum = (
        metric[layer_idx * 7]  + #Q
        metric[layer_idx * 7 + 1] + #K
        metric[layer_idx * 7 + 2] + #V
        metric[layer_idx * 7 + 3]   #OUT 
        #metric[layer_idx * 7 + 4] + #GATE
        #metric[layer_idx * 7 + 5] +  #UP
        #metric[layer_idx * 7 + 6]  #DOWN
    )
    
   
    print(attn_hill_sum/4)
    return layer_idx, attn_hill_sum/4


# 🚀 计算所有层的重要性
lambda_esd = 1  # 可以调整这个参数
layer_importance_scores = [process_layer(idx, layer) for idx, layer in enumerate(model.model.layers)]

# 🚀 归一化
scores = torch.tensor([imp[1] for imp in layer_importance_scores])
s1, s2 = 0.8, 1.2
max_score, min_score = scores.max(), scores.min()
normalized_scores = ((scores - min_score) / (max_score - min_score)) * (s2 - s1) + s1

# 调整均值到 0.7
scale = 0.7 / normalized_scores.mean()
normalized_scores = normalized_scores * scale
print(normalized_scores.mean())
# 打印最终结果
print("\n🔝 LLaMA 7B 每层的归一化相对重要性:")
res = []
for (idx, _), importance in zip(layer_importance_scores, normalized_scores.tolist()):
    print(f"Layer {idx}: {importance:.4f}")
    res.append(importance)
print(res)

