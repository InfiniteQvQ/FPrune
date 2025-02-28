import torch
import weightwatcher as ww
from transformers import AutoModelForCausalLM

def analyze_layer_weightwatcher(layer, layer_idx):
    """使用 weightwatcher 计算 PL_Alpha_Hill 和 KS 统计量"""
    watcher = ww.WeightWatcher()
    details = watcher.analyze(layer, min_evals=3, randomize=False)  # 自动计算所有权重的幂律指数
    
    # 获取幂律指数和 KS 统计量
    alpha_hill = details["alpha"].mean()  # 取平均幂律指数
    ks_stat = details["lognorm"].mean()  # 取均值 KS 统计量（log 正态偏差）
    
    print(f"Layer {layer_idx}: PL_Alpha_Hill={alpha_hill:.4f}, KS D={ks_stat:.4f}")
    return alpha_hill, ks_stat

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
