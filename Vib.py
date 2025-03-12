import torch
import numpy as np
import joblib  # ✅ 并行计算
from transformers import AutoModelForCausalLM

# 🛠️ 设置 PyTorch 线程数（可调整）
torch.set_num_threads(8)  # ✅ 设置最大使用 8 个 CPU 线程

# 🛠️ 设置缓存目录
cache_dir = "/root/autodl-tmp/llm_weights"

# 🚀 加载 LLaMA-7B（只在 CPU 运行）
model = AutoModelForCausalLM.from_pretrained(
    "pinkmanlove/llama-7b-hf",
    cache_dir=cache_dir,
    device_map="auto",   # ✅ 强制使用 CPU
    torch_dtype=torch.float32,  # ✅ 保持计算精度
)

# 🎯 计算 SVD 奇异值谱（多进程并行）
def singular_value_spectrum(weight_matrix):
    """计算 SVD 奇异值谱"""
    weight_matrix = weight_matrix.float()  # 避免 float16 报错
    with torch.no_grad():
        singular_values = torch.linalg.svdvals(weight_matrix)  # ✅ 只计算奇异值，加速
    return np.sum(singular_values.cpu().numpy())  # 直接返回求和，加速计算

# 🎯 计算特征值谱分布 (ESD)（多进程并行）
def esd_spectrum(weight_matrix):
    """计算特征值谱分布 (ESD)"""
    weight_matrix = weight_matrix.float()
    with torch.no_grad():
        max_eigval = torch.linalg.eigvalsh(weight_matrix @ weight_matrix.T).max()
    return max_eigval.cpu().numpy()  # 直接返回最大特征值，加速计算

# 🎯 计算单层重要性（并行执行）
def process_layer(layer_idx, layer):
    print(f"Processing Layer {layer_idx}...")

    # 🧠 Attention 层（用 SVD）
    q_proj = layer.self_attn.q_proj.weight
    k_proj = layer.self_attn.k_proj.weight
    v_proj = layer.self_attn.v_proj.weight
    attn_score = np.mean([
        singular_value_spectrum(q_proj),
        singular_value_spectrum(k_proj),
        singular_value_spectrum(v_proj)
    ])  # SVD 计算重要性

    # 🔥 MLP 层（用 ESD）
    gate_proj = layer.mlp.gate_proj.weight
    up_proj = layer.mlp.up_proj.weight
    down_proj = layer.mlp.down_proj.weight
    mlp_score = np.mean([
        esd_spectrum(gate_proj),
        esd_spectrum(up_proj),
        esd_spectrum(down_proj)
    ])  # ESD 计算重要性

    # 🎯 Output 层（用 SVD）
    output_proj = layer.self_attn.o_proj.weight
    output_score = singular_value_spectrum(output_proj)  # SVD 计算重要性

    # 📊 计算相对重要性
    layer_relative_importance = attn_score + mlp_score + output_score

    return layer_idx, layer_relative_importance  # ✅ 返回结果

# 🚀 并行计算每一层
layer_importance_scores = joblib.Parallel(n_jobs=-1)(
    joblib.delayed(process_layer)(idx, layer) for idx, layer in enumerate(model.model.layers)
)

# 🚀 排序
sorted_layers = sorted(layer_importance_scores, key=lambda x: x[1], reverse=True)

print("\n🔝 LLaMA 7B 每层的相对重要性排序:")
for idx, importance in sorted_layers:
    print(f"Layer {idx}: {importance:.4f}")
