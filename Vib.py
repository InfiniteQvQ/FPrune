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
def pl_alpha_hill_peak(weight_matrix, bins=100):
    """使用 'xmin_peak' 方法计算 PL_Alpha_Hill"""
    weight_matrix = weight_matrix.float()
    with torch.no_grad():
        eigvals = torch.linalg.eigvalsh(weight_matrix @ weight_matrix.T).cpu().numpy()
    eigvals = np.sort(eigvals)  # 升序排列

    # 计算 log-scale 直方图，找到最大密度点作为 xmin
    log_nz_eigs = np.log10(eigvals[eigvals > 0])  # 过滤掉零值
    min_e, max_e = log_nz_eigs.min(), log_nz_eigs.max()
    counts, bin_edges = np.histogram(log_nz_eigs, bins=bins, range=(min_e, max_e))
    peak_idx = np.argmax(counts)  # 找到峰值
    xmin = 10 ** bin_edges[peak_idx]  # 还原回非 log 值

    # 计算 Hill 估计值
    valid_eigs = eigvals[eigvals >= xmin]
    n = len(valid_eigs)
    if n < 2: return 1.0  # 避免除零错误
    alpha_hill = 1 + n / (np.sum(np.log(valid_eigs / xmin)))

    return alpha_hill

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
        pl_alpha_hill_peak(layer.self_attn.q_proj.weight) +
        pl_alpha_hill_peak(layer.self_attn.k_proj.weight) +
        pl_alpha_hill_peak(layer.self_attn.v_proj.weight) +
        pl_alpha_hill_peak(layer.self_attn.o_proj.weight) + 
        pl_alpha_hill_peak(layer.mlp.gate_proj.weight) + 
        pl_alpha_hill_peak(layer.mlp.up_proj.weight) + 
        pl_alpha_hill_peak(layer.mlp.down_proj.weight) 
    )

   
    print(attn_hill_sum/7)
    return layer_idx, attn_hill_sum


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

