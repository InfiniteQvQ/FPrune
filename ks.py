import torch
import powerlaw
import numpy as np
from transformers import AutoModelForCausalLM

def hill_estimator(weights, k=50):
    """计算 PL_Alpha_Hill 幂律指数，支持 GPU"""
    sorted_weights = torch.sort(torch.abs(weights), descending=True)[0]  # 降序排列
    x_k = sorted_weights[k-1]
    alpha = 1 + (1 / torch.mean(torch.log(sorted_weights[:k] / x_k)))
    return alpha.item()

def ks_test_powerlaw(weights):
    """计算 PL_Alpha_KS 统计量，优化 GPU 计算"""
    weights_np = weights.detach().cpu().float().numpy().flatten()  # 只在需要时拷贝
    fit = powerlaw.Fit(weights_np, verbose=False)
    return fit.alpha, fit.D  # 返回幂律指数 α 和 KS 统计量 D

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
