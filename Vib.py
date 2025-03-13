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
    """
    使用 'xmin_peak' 方法计算 PL_Alpha_Hill（alphahill）的值

    参数：
      weight_matrix: 权重矩阵（例如 layer.self_attn.q_proj.weight）
      bins: 用于直方图的箱数（默认 100）

    返回：
      final_alphahat: 归一化后的 alphahill 数值
    """
    weight_matrix = weight_matrix.float()
    with torch.no_grad():
        # 计算 matrix @ matrix.T 的特征值，得到实数特征值
        eigvals = torch.linalg.eigvalsh(weight_matrix @ weight_matrix.T).cpu().numpy()

    # 将特征值按升序排列
    eigvals = np.sort(eigvals)

    # 过滤掉零值并取对数，构造 log-scale 直方图
    positive_eigs = eigvals[eigvals > 0]
    if len(positive_eigs) == 0:
        return 1.0  # 如果没有正特征值，返回默认值
    log_nz_eigs = np.log10(positive_eigs)
    min_e, max_e = log_nz_eigs.min(), log_nz_eigs.max()

    # 构造直方图并选择直方图密度最大的箱对应的 xmin
    counts, bin_edges = np.histogram(log_nz_eigs, bins=bins, range=(min_e, max_e))
    peak_idx = np.argmax(counts)
    xmin = 10 ** bin_edges[peak_idx]

    # 设置 xmin 的限制范围，避免极端情况
    xmin_min = 0.95 * xmin
    xmin_max = 1.5 * xmin

    # 筛选出处于 [xmin, xmin_max] 范围内的特征值
    valid_eigs = eigvals[(eigvals >= xmin) & (eigvals <= xmin_max)]
    n = len(valid_eigs)
    if n < 2:
        return 1.0  # 特征值太少时返回默认值

    # 遍历不同候选 xmin 值，计算对应的 alpha 和拟合指标 D
    alphas = []
    Ds = []
    for i, current_xmin in enumerate(valid_eigs[:-1]):
        tail = valid_eigs[i:]
        alpha = 1 + len(tail) / np.sum(np.log(tail / current_xmin))
        alphas.append(alpha)
        D = np.max(np.abs(1 - (tail / current_xmin) ** (-alpha + 1) - np.arange(len(tail)) / len(tail)))
        Ds.append(D)

    # 选择使 D 最小的 alpha
    min_D_index = np.argmin(Ds)
    final_alpha = alphas[min_D_index]

    # 使用谱范数归一化得到最终 alphahill
    spectral_norm = np.max(eigvals)
    final_alphahat = final_alpha * np.log10(spectral_norm)

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

