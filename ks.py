import torch
import weightwatcher as ww
from transformers import AutoModelForCausalLM

def analyze_layer_weightwatcher(layer, layer_idx):
    """使用 weightwatcher 计算 KS 统计量"""
    watcher = ww.WeightWatcher()
    details = watcher.analyze(layer, min_evals=3, randomize=False)  # 自动分析权重
    
    # 打印所有可用列，确保 log_norm 存在
    print(f"Layer {layer_idx} Details:\n", details.columns)
    
    # 计算 KS 统计量（归一化处理）
    if "log_norm" in details.columns:
        ks_stat = details["log_norm"].mean()
        ks_stat = ks_stat / details["log_norm"].max()  # 归一化
    else:
        ks_stat = float('nan')

    print(f"Layer {layer_idx}: KS D={ks_stat:.4f}")
    return float(ks_stat)  # 只返回 KS 统计量

def analyze_llama7b(model_name="meta-llama/Llama-7b-hf", device="cuda"):
    """计算 LLaMA-7B 每层的 KS 统计量（用 weightwatcher）"""
    
    cache_dir = "/root/autodl-tmp/llm_weights"
    model = AutoModelForCausalLM.from_pretrained(
        "pinkmanlove/llama-7b-hf",
        cache_dir=cache_dir,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    ks_stats = []  # 仅存储 KS 统计量的列表

    for layer_idx, layer in enumerate(model.model.layers):  # 遍历 LLaMA Transformer 层
        ks_stat = analyze_layer_weightwatcher(layer, layer_idx)
        ks_stats.append(ks_stat)

    return ks_stats  # 仅返回 KS 统计量的列表

if __name__ == "__main__":
    ks_stats = analyze_llama7b(device="cuda")  # 使用 GPU 计算

    # 打印最终 KS 统计量列表
    print("\n🔥 Final KS Statistics (All Layers):")
    print(ks_stats)
