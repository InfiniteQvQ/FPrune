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

    # 设定 xmin 限制范围，避免极端情况
    xmin_min = 10 ** np.log10(0.95 * xmin)
    xmin_max = 1.5 * xmin

    # 限制 eigvals 范围
    valid_eigs = eigvals[(eigvals >= xmin) & (eigvals <= xmin_max)]
    n = len(valid_eigs)
    if n < 2: return 1.0  # 避免除零错误

    # 遍历不同 xmin 选择最优 alpha
    alphas = []
    Ds = []
    for i, xmin in enumerate(valid_eigs[:-1]):
        alpha = 1 + len(valid_eigs[i:]) / np.sum(np.log(valid_eigs[i:] / xmin))
        alphas.append(alpha)
        D = np.max(np.abs(1 - (valid_eigs[i:] / xmin) ** (-alpha + 1) - np.arange(len(valid_eigs[i:])) / len(valid_eigs[i:])))
        Ds.append(D)

    min_D_index = np.argmin(Ds)  # 选择 D 最小的 alpha
    final_alpha = alphas[min_D_index]

    # 计算 spectral norm 归一化的 alpha
    spectral_norm = np.max(eigvals)  # 获取谱范数
    final_alphahat = final_alpha * np.log10(spectral_norm)  # 归一化

    return final_alphahat


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
    
   
    print(attn_hill_sum)
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

