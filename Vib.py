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

# 🎯 计算 PL_Alpha_Hill
def pl_alpha_hill_peak(weight_matrix, bins=100, EVALS_THRESH=0.00001, conv_norm=0.5, filter_zeros=False):
    """
    使用与 net_esd_estimator (fix_fingers='xmin_peak') 完全一致的方式计算 PL_Alpha_Hill（alphahill）的值。
    
    参数:
      weight_matrix: 待计算的权重矩阵（例如 layer.self_attn.q_proj.weight）
      bins: 直方图箱数，默认 100
      EVALS_THRESH: 过滤近零特征值的阈值，默认 0.00001
      conv_norm: 针对 Conv2d 层归一化因子，默认 0.5（注意如果传入的是 Linear 层权重，此参数可能不需要）
      filter_zeros: 是否过滤小于阈值的特征值，默认 False
      
    返回:
      final_alphahat: 归一化后的 alphahill 数值，计算方式为 final_alpha * log10(spectral_norm)
    """
    weight_matrix = weight_matrix.float()
    
    # 对于 pl_alpha_hill_peak，我们直接计算 weight_matrix @ weight_matrix.T 的特征值
    with torch.no_grad():
        eigvals = torch.linalg.eigvalsh(weight_matrix @ weight_matrix.T).cpu()
    
    # 将特征值升序排列
    eigvals, _ = torch.sort(eigvals, descending=False)
    spectral_norm = eigvals[-1].item()      # 最大特征值
    fnorm = torch.sum(eigvals).item()         # 所有特征值和（如需要）
    
    # 过滤接近零的特征值（如果设置了过滤）
    if filter_zeros:
        nz_eigs = eigvals[eigvals > EVALS_THRESH]
        N = len(nz_eigs)
        if N == 0:
            nz_eigs = eigvals
            N = len(nz_eigs)
    else:
        nz_eigs = eigvals
        N = len(nz_eigs)
    
    # 计算对数（用于后续拟合计算）——注意：alpha计算用的是自然对数
    log_nz_eigs = torch.log(nz_eigs)
    
    # 构造 log10 直方图，选择密度峰值对应的 xmin
    hist_nz_eigs = torch.log10(nz_eigs)
    min_e, max_e = hist_nz_eigs.min(), hist_nz_eigs.max()
    counts = torch.histc(hist_nz_eigs, bins=bins, min=min_e, max=max_e)
    boundaries = torch.linspace(min_e, max_e, bins + 1)
    ih = torch.argmax(counts).item()
    xmin2 = 10 ** boundaries[ih].item()
    
    # 设置 xmin 限制范围（注意：下界这里取的是 log10(0.95*xmin2)）
    xmin_min = torch.log10(0.95 * xmin2)
    xmin_max = 1.5 * xmin2
    
    # 遍历候选的 xmin 值，计算 alpha 及拟合指标 D
    alphas = torch.zeros(N - 1)
    Ds = torch.ones(N - 1)
    for i, xmin in enumerate(nz_eigs[:-1]):
        # 采用与 net_esd_estimator 中 'xmin_peak' 分支一致的过滤：
        # 注意：这里直接比较原始值 xmin 与 xmin_max，而 xmin_min 为 log10 值，
        # 原代码中即为：if xmin < xmin_min: continue
        if xmin < xmin_min:
            continue
        if xmin > xmin_max:
            break
        
        n = float(N - i)
        seq = torch.arange(n, dtype=torch.float32)
        # 计算 alpha = 1 + n / (sum(log(candidate_tail)) - n * log(candidate))
        alpha = 1 + n / (torch.sum(log_nz_eigs[i:]) - n * log_nz_eigs[i])
        alphas[i] = alpha
        if alpha > 1:
            D = torch.max(torch.abs(1 - (nz_eigs[i:] / xmin) ** (-alpha + 1) - seq / n))
            Ds[i] = D
    
    min_D_index = torch.argmin(Ds).item()
    final_alpha = alphas[min_D_index].item()
    
    # 使用谱范数的 log10 对 alpha 归一化，得到最终的 alphahill 值
    final_alphahat = final_alpha * math.log10(spectral_norm)
    
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

