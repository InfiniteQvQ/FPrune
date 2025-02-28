import torch
import powerlaw
import numpy as np
from transformers import AutoModelForCausalLM

def hill_estimator(weights, k=50):
    """计算 PL_Alpha_Hill 幂律指数"""
    sorted_weights = np.sort(weights)[::-1]  # 降序排列
    x_k = sorted_weights[k-1]
    alpha = 1 + (1 / np.mean(np.log(sorted_weights[:k] / x_k)))
    return alpha

def ks_test_powerlaw(weights):
    """计算 PL_Alpha_KS 统计量"""
    fit = powerlaw.Fit(weights)
    return fit.alpha, fit.D  # 返回幂律指数 α 和 KS 统计量 D

def analyze_layer(layer_weights, layer_idx):
    """计算单层 Transformer 的 PL_Alpha_Hill 和 KS 统计量"""
    weights = layer_weights.detach().cpu().numpy().flatten()
    alpha_hill = hill_estimator(weights)
    alpha_ks, ks_stat = ks_test_powerlaw(weights)
    print(f"Layer {layer_idx}: PL_Alpha_Hill={alpha_hill:.4f}, PL_Alpha_KS={alpha_ks:.4f}, KS D={ks_stat:.4f}")
    return alpha_hill, alpha_ks, ks_stat

def analyze_llama7b(model_name="meta-llama/Llama-7b-hf"):
    """计算 LLaMA-7B 每层及整体是否符合幂律剪枝"""
    
    cache_dir = "/root/autodl-tmp/llm_weights"
    model = AutoModelForCausalLM.from_pretrained(
        "pinkmanlove/llama-7b-hf",
        cache_dir=cache_dir,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    all_weights = []
    results = []
    
    for layer_idx, layer in enumerate(model.model.layers):  # LLaMA Transformer 层
        if hasattr(layer, 'self_attn'):
            weights = layer.self_attn.q_proj.weight  # 选择 Query 权重
            alpha_hill, alpha_ks, ks_stat = analyze_layer(weights, layer_idx)
            results.append((layer_idx, alpha_hill, alpha_ks, ks_stat))
            all_weights.extend(weights.detach().cpu().numpy().flatten())
    
    # 计算整个模型的 PL_Alpha_KS
    print("\nAnalyzing overall model...")
    overall_alpha_hill = hill_estimator(np.array(all_weights))
    overall_alpha_ks, overall_ks_stat = ks_test_powerlaw(np.array(all_weights))
    print(f"Overall Model: PL_Alpha_Hill={overall_alpha_hill:.4f}, PL_Alpha_KS={overall_alpha_ks:.4f}, KS D={overall_ks_stat:.4f}")
    
    return results, (overall_alpha_hill, overall_alpha_ks, overall_ks_stat)

if __name__ == "__main__":
    results, overall = analyze_llama7b()
    print("\nFinal Summary:")
    for layer in results:
        print(f"Layer {layer[0]}: PL_Alpha_Hill={layer[1]:.4f}, PL_Alpha_KS={layer[2]:.4f}, KS D={layer[3]:.4f}")
    print(f"\nOverall Model: PL_Alpha_Hill={overall[0]:.4f}, PL_Alpha_KS={overall[1]:.4f}, KS D={overall[2]:.4f}")
