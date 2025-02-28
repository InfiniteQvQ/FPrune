import torch
import numpy as np
from transformers import AutoModelForCausalLM

def hill_estimator(weights, k=50, epsilon=1e-8):
    """计算 PL_Alpha_Hill 幂律指数，优化 NaN 处理"""
    sorted_weights = torch.sort(torch.abs(weights), descending=True)[0]  # 降序排列
    sorted_weights = sorted_weights[sorted_weights > epsilon]  # 过滤掉接近 0 的值

    if len(sorted_weights) < k:  
        return float('nan')  # 如果数据不足，返回 NaN
    
    x_k = sorted_weights[k-1]  # 选取第 k 大的权重
    x_k = max(x_k, epsilon)  # 确保 x_k 不为 0

    alpha = 1 + (1 / torch.mean(torch.log(sorted_weights[:k] / x_k)))
    return alpha.item()

def ks_test_powerlaw(weights, epsilon=1e-8):
    """优化 KS 统计量计算，避免 NaN"""
    weights_np = weights.detach().cpu().float().numpy().flatten()
    weights_np = weights_np[weights_np > epsilon]  # 过滤掉接近 0 的值
    
    if len(weights_np) < 2:
        return float('nan'), float('nan')  # 避免 KS 计算无效
    
    # 经验分布函数 (ECDF)
    empirical_cdf = np.sort(weights_np)
    empirical_cdf = np.arange(1, len(empirical_cdf) + 1) / len(empirical_cdf)
    
    # 理论幂律分布 CDF
    min_weight = np.min(weights_np)
    power_cdf = (weights_np / min_weight) ** -1.5  # 近似幂律
    power_cdf /= np.max(power_cdf)

    # 计算 KS 统计量
    ks_stat = np.max(np.abs(empirical_cdf - power_cdf))
    
    # 估计幂律指数 α
    alpha_ks = 1 + (np.mean(np.log(weights_np / min_weight))) ** -1

    return alpha_ks, ks_stat

def analyze_layer(layer, layer_idx, device):
    """计算单层 Transformer 的 7 个矩阵的 PL_Alpha_Hill 和 KS 统计量"""
    matrices = {
        "Q": layer.self_attn.q_proj.weight,
        "K": layer.self_attn.k_proj.weight,
        "V": layer.self_attn.v_proj.weight,
        "Out": layer.self_attn.o_proj.weight,
        "Gate": layer.mlp.gate_proj.weight,
        "Up": layer.mlp.up_proj.weight,
        "Down": layer.mlp.down_proj.weight
    }
    
    results = {}
    
    for name, weights in matrices.items():
        weights = weights.to(device)  # 直接在 GPU 计算
        alpha_hill = hill_estimator(weights)
        alpha_ks, ks_stat = ks_test_powerlaw(weights)
        results[name] = (alpha_hill, alpha_ks, ks_stat)
        print(f"Layer {layer_idx} - {name}: PL_Alpha_Hill={alpha_hill:.4f}, PL_Alpha_KS={alpha_ks:.4f}, KS D={ks_stat:.4f}")
    
    return results

def analyze_llama7b(model_name="meta-llama/Llama-7b-hf", device="cuda"):
    """计算 LLaMA-7B 每层及整体是否符合幂律剪枝"""
    
    cache_dir = "/root/autodl-tmp/llm_weights"
    model = AutoModelForCausalLM.from_pretrained(
        "pinkmanlove/llama-7b-hf",
        cache_dir=cache_dir,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    results = []
    
    for layer_idx, layer in enumerate(model.model.layers):  # 遍历 LLaMA Transformer 层
        layer_results = analyze_layer(layer, layer_idx, device)
        results.append((layer_idx, layer_results))

    return results

if __name__ == "__main__":
    results = analyze_llama7b(device="cuda")  # 使用 GPU 计算
    print("\nFinal Summary:")
    for layer in results:
        layer_idx, layer_results = layer
        print(f"\nLayer {layer_idx} Summary:")
        for name, (alpha_hill, alpha_ks, ks_stat) in layer_results.items():
            print(f"  {name}: PL_Alpha_Hill={alpha_hill:.4f}, PL_Alpha_KS={alpha_ks:.4f}, KS D={ks_stat:.4f}")
