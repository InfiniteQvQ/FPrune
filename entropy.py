import torch
import numpy as np
from transformers import AutoModelForCausalLM

# **获取 GPU 设备**
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🎯 计算 Spectral Entropy（衡量信息密度）
def compute_spectral_entropy(weight_matrix):
    eigs = torch.linalg.svdvals(weight_matrix).pow(2)
    eigs = eigs / eigs.sum()
    entropy = -torch.sum(eigs * torch.log(eigs + 1e-9))
    return entropy.item()

# 🎯 计算 Spectral Norm（衡量权重矩阵的重要性）
def compute_spectral_norm(weight_matrix):
    return torch.linalg.svdvals(weight_matrix).max().item()

# 🎯 计算 Gradient Norm（衡量梯度对模型的影响）
def compute_gradient_norm(weight_matrix):
    return torch.norm(weight_matrix.grad).item() if weight_matrix.grad is not None else 0.0

# 🚀 计算每个模块的 Entropy / Norm / Gradient Norm
def analyze_layer(layer):
    results = {}

    # **确保权重矩阵在 GPU 上计算**
    results["QKV_Entropy"] = (
        compute_spectral_entropy(layer.self_attn.q_proj.weight.to(device)) +
        compute_spectral_entropy(layer.self_attn.k_proj.weight.to(device)) +
        compute_spectral_entropy(layer.self_attn.v_proj.weight.to(device))
    )
    results["QKV_Norm"] = (
        compute_spectral_norm(layer.self_attn.q_proj.weight.to(device)) +
        compute_spectral_norm(layer.self_attn.k_proj.weight.to(device)) +
        compute_spectral_norm(layer.self_attn.v_proj.weight.to(device))
    )
    results["QKV_Grad"] = (
        compute_gradient_norm(layer.self_attn.q_proj.weight) +
        compute_gradient_norm(layer.self_attn.k_proj.weight) +
        compute_gradient_norm(layer.self_attn.v_proj.weight)
    )

    results["Output_Entropy"] = compute_spectral_entropy(layer.self_attn.o_proj.weight.to(device))
    results["Output_Norm"] = compute_spectral_norm(layer.self_attn.o_proj.weight.to(device))
    results["Output_Grad"] = compute_gradient_norm(layer.self_attn.o_proj.weight)

    results["Gate_Entropy"] = compute_spectral_entropy(layer.mlp.gate_proj.weight.to(device))
    results["Gate_Norm"] = compute_spectral_norm(layer.mlp.gate_proj.weight.to(device))
    results["Gate_Grad"] = compute_gradient_norm(layer.mlp.gate_proj.weight)

    results["Up_Entropy"] = compute_spectral_entropy(layer.mlp.up_proj.weight.to(device))
    results["Up_Norm"] = compute_spectral_norm(layer.mlp.up_proj.weight.to(device))
    results["Up_Grad"] = compute_gradient_norm(layer.mlp.up_proj.weight)

    results["Down_Entropy"] = compute_spectral_entropy(layer.mlp.down_proj.weight.to(device))
    results["Down_Norm"] = compute_spectral_norm(layer.mlp.down_proj.weight.to(device))
    results["Down_Grad"] = compute_gradient_norm(layer.mlp.down_proj.weight)

    return results

# 🚀 处理整个模型
def process_model(model_name, model_path):
    print(f"\n🚀 Loading {model_name} on {device}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        cache_dir="/root/autodl-tmp/llm_weights",
        device_map="auto",
        torch_dtype=torch.float16
    ).to(device)  # **加载模型到 GPU**

    all_results = {
        "QKV_Entropy": [], "QKV_Norm": [], "QKV_Grad": [],
        "Output_Entropy": [], "Output_Norm": [], "Output_Grad": [],
        "Gate_Entropy": [], "Gate_Norm": [], "Gate_Grad": [],
        "Up_Entropy": [], "Up_Norm": [], "Up_Grad": [],
        "Down_Entropy": [], "Down_Norm": [], "Down_Grad": []
    }
    
    for idx, layer in enumerate(model.model.layers):
        print(f"Processing Layer {idx}...")
        layer_results = analyze_layer(layer)
        for key in all_results:
            all_results[key].append(layer_results[key])

    # 🔥 清空缓存
    del model
    torch.cuda.empty_cache()
    print("\n🧹 CUDA Cache Cleared!\n")

    return {key: np.array(values) for key, values in all_results.items()}

# 🚀 计算 LLaMA 2 (7B) 和 LLaMA 7B 的模块影响
llama2_7b_results = process_model("LLaMA 2 (7B)", "meta-llama/Llama-2-7b-hf")
llama7b_results = process_model("LLaMA 7B)", "pinkmanlove/llama-7b-hf")

# 🚀 计算各模块的平均影响
for key in llama2_7b_results:
    print(f"\n🔍 **{key} Comparison**")
    print(f"LLaMA 2 (7B) {key} (Avg): {np.mean(llama2_7b_results[key]):.4f}")
    print(f"LLaMA 7B {key} (Avg): {np.mean(llama7b_results[key]):.4f}")
