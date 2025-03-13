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
def compute_alpha_peak(weight_matrix, bins=100, EVALS_THRESH=1e-5, conv_norm=0.5, filter_zeros=False):
    """
    根据 net_esd_estimator 中 fix_fingers='xmin_peak' 分支的逻辑，
    对单个权重矩阵计算 alpha_peak 值，并返回 final_alphahat（即 final_alpha * log10(spectral_norm)）。
    
    参数:
      weight_matrix: 权重矩阵（来自 nn.Conv2d 或 nn.Linear），如果是 Conv2d 权重，则进行扁平化和归一化处理。
      bins: 直方图箱数，默认 100。
      EVALS_THRESH: 过滤近零特征值的阈值，默认 1e-5。
      conv_norm: 针对 Conv2d 层的归一化因子，默认 0.5（对于 Linear 层一般无需调整）。
      filter_zeros: 是否过滤小于阈值的特征值，默认 False。
      
    返回:
      final_alphahat: 根据最终选择的 alpha 归一化得到的值，与 net_esd_estimator 的计算完全一致。
    """
    # 如果权重矩阵维度大于2，则认为其来自 Conv2d，需要先扁平化和归一化
    if weight_matrix.ndim > 2:
        matrix = weight_matrix.clone().cpu()
        matrix = torch.flatten(matrix, start_dim=2) * math.sqrt(conv_norm)
        matrix = matrix.transpose(1, 2).transpose(0, 1)
    else:
        matrix = weight_matrix.clone().cpu()
    matrix = matrix.to(torch.float32)
    
    # 计算特征值：对 weight 矩阵计算 SVD，然后对 singular values 平方
    eigs = torch.square(torch.linalg.svdvals(matrix).flatten())
    eigs, _ = torch.sort(eigs, descending=False)
    spectral_norm = eigs[-1].item()
    
    # 过滤或不过滤近零特征值
    if filter_zeros:
        nz_eigs = eigs[eigs > EVALS_THRESH]
        N = nz_eigs.numel()
        if N == 0:
            nz_eigs = eigs
            N = nz_eigs.numel()
    else:
        nz_eigs = torch.clamp(eigs, min=EVALS_THRESH)
        N = nz_eigs.numel()
    
    # 计算自然对数，用于后续 alpha 计算
    log_nz_eigs = torch.log(nz_eigs)
    
    # 利用 log10 计算直方图，选择密度最大的箱对应的 xmin2
    hist_nz_eigs = torch.log10(nz_eigs)
    min_e, max_e = hist_nz_eigs.min(), hist_nz_eigs.max()
    counts = torch.histc(hist_nz_eigs, bins=bins, min=min_e.item(), max=max_e.item())
    boundaries = torch.linspace(min_e.item(), max_e.item(), bins + 1)
    ih = torch.argmax(counts).item()
    xmin2 = 10 ** boundaries[ih].item()
    
    # 设置 xmin 限制范围
    xmin_min = torch.log10(torch.tensor(0.95 * xmin2))
    xmin_max = 1.5 * xmin2
    
    # 遍历候选的 xmin 值，计算对应的 alpha 和拟合指标 D
    alphas = torch.zeros(N - 1)
    Ds = torch.ones(N - 1)
    for i, xmin in enumerate(nz_eigs[:-1]):
        if xmin < xmin_min:
            continue
        if xmin > xmin_max:
            break
        
        n = float(N - i)
        seq = torch.arange(n, dtype=torch.float32)
        alpha = 1 + n / (torch.sum(log_nz_eigs[i:]) - n * log_nz_eigs[i])
        alphas[i] = alpha
        if alpha > 1:
            D = torch.max(torch.abs(1 - (nz_eigs[i:] / xmin) ** (-alpha + 1) - seq / n))
            Ds[i] = D
    
    min_D_index = torch.argmin(Ds).item()
    final_alpha = alphas[min_D_index].item()
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
        compute_alpha_peak(layer.self_attn.q_proj.weight) +
        compute_alpha_peak(layer.self_attn.k_proj.weight) +
        compute_alpha_peak(layer.self_attn.v_proj.weight) +
        compute_alpha_peak(layer.self_attn.o_proj.weight) + 
        compute_alpha_peak(layer.mlp.gate_proj.weight) + 
        compute_alpha_peak(layer.mlp.up_proj.weight) + 
        compute_alpha_peak(layer.mlp.down_proj.weight) 
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

