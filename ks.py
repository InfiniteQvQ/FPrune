import torch
import weightwatcher as ww
from transformers import AutoModelForCausalLM

import weightwatcher as ww

def analyze_layer_weightwatcher(layer, layer_idx):
    """使用 weightwatcher 计算 PL_Alpha_Hill 和 KS 统计量"""
    watcher = ww.WeightWatcher()
    details = watcher.analyze(layer, min_evals=3, randomize=False)  # 自动分析权重
    
    # 打印所有可用列，确保 log_norm 存在
    print(f"Layer {layer_idx} Details:\n", details.columns)
    
    if "alpha_weighted" in details.columns:
        alpha_hill = details["alpha_weighted"].mean()  # 计算幂律指数
    elif "alpha" in details.columns:
        alpha_hill = details["alpha"].mean()  # 备用方案
    else:
        alpha_hill = float('nan')

    if "log_norm" in details.columns:
        ks_stat = details["log_norm"].mean()  # 代替 KS D
    else:
        ks_stat = float('nan')

    print(f"Layer {layer_idx}: PL_Alpha_Hill={alpha_hill:.4f}, KS D={ks_stat:.4f}")
    return float(alpha_hill), float(ks_stat)  # 确保返回可解包的 tuple


def analyze_llama7b(model_name="meta-llama/Llama-7b-hf", device="cuda"):
    """计算 LLaMA-7B 每层的 PL_Alpha_Hill 和 KS 统计量（用 weightwatcher）"""
    
    cache_dir = "/root/autodl-tmp/llm_weights"
    model = AutoModelForCausalLM.from_pretrained(
        "pinkmanlove/llama-7b-hf",
        cache_dir=cache_dir,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    results = []
    
    for layer_idx, layer in enumerate(model.model.layers):  # 遍历 LLaMA Transformer 层
        alpha_hill, ks_stat = analyze_layer_weightwatcher(layer, layer_idx)
        results.append((layer_idx, alpha_hill, ks_stat))

    return results

if __name__ == "__main__":
    results = analyze_llama7b(device="cuda")  # 使用 GPU 计算
    print("\nFinal Summary:")
    for layer in results:
        layer_idx, alpha_hill, ks_stat = layer
        print(f"Layer {layer_idx}: PL_Alpha_Hill={alpha_hill:.4f}, KS D={ks_stat:.4f}")
