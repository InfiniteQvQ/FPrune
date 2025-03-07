import torch
import numpy as np
from transformers import LlamaModel, AutoTokenizer

def compute_KTA_approx(H, T, sample_size=100):
    """
    采用采样方式计算 KTA，减少矩阵尺寸，从而降低数值溢出风险。
    
    Args:
        H (numpy.ndarray): 形状 (seq_len, hidden_dim) 的隐藏状态矩阵
        T (numpy.ndarray): 形状 (seq_len, seq_len) 的目标矩阵（通常由标签生成）
        sample_size (int): 采样的 token 数量
    Returns:
        float: 近似计算得到的 KTA 值
    """
    num_tokens = H.shape[0]
    sample_idx = np.random.choice(num_tokens, size=min(sample_size, num_tokens), replace=False)
    H_sample = H[sample_idx, :]
    T_sample = T[np.ix_(sample_idx, sample_idx)]
    
    # 将采样的 H 转为高精度 float64
    H_sample = H_sample.astype(np.float64)
    
    # 归一化 H_sample，每个 token 向量除以它的范数，防止数值过大
    norm_H = np.linalg.norm(H_sample, axis=1, keepdims=True) + 1e-6
    H_sample = H_sample / norm_H
    # 限制值域，防止极端值影响矩阵乘法
    H_sample = np.clip(H_sample, -1, 1)
    
    # 计算核矩阵 K_sample
    K_sample = np.matmul(H_sample, H_sample.T)
    # 同样对 T_sample 做归一化（如果 T 由标签构成，通常取值较小，但仍处理一下）
    T_sample = T_sample.astype(np.float64)
    norm_T = np.linalg.norm(T_sample, axis=1, keepdims=True) + 1e-6
    T_sample = T_sample / norm_T
    
    # 计算 Frobenius 范数（加 epsilon 防止 0 除法）
    K_norm = np.sqrt(np.sum(K_sample**2)) + 1e-6
    T_norm = np.sqrt(np.sum(T_sample**2)) + 1e-6
    inner_product = np.sum(K_sample * T_sample)
    
    return inner_product / (K_norm * T_norm)


# 设置设备，这里使用 device_map="auto" 自动分配多GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "pinkmanlove/llama-7b-hf"
# 使用 LlamaModel 而非 LlamaForCausalLM，避免 lm_head 警告
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceM4/llama-7b-tokenizer")
model = LlamaModel.from_pretrained(
    model_name,
    cache_dir="/root/autodl-tmp/llm_weights",
    output_hidden_states=True,
    torch_dtype=torch.float16,
    device_map="auto"
)

# 处理输入，移除 token_type_ids（Llama 不需要）
text = ["LLaMA 7B Kernel Target Alignment computation."]
inputs = tokenizer(text, return_tensors="pt")
inputs.pop("token_type_ids", None)
inputs = {key: val.to(device) for key, val in inputs.items()}

num_runs = 10  # 多次运行取平均
kta_results = {i: [] for i in range(model.config.num_hidden_layers)}

for run in range(num_runs):
    with torch.no_grad():
        outputs = model(**inputs)
    hidden_states = outputs.hidden_states  # tuple of (num_layers, batch, seq_len, hidden_dim)
    seq_len = hidden_states[0].shape[1]
    # 使用固定随机种子确保采样一致性
    np.random.seed(42)
    # 这里假设目标矩阵 T 由随机标签生成。若有实际标签，请替换以下代码：
    labels = np.random.choice([-1, 1], size=(seq_len, 1))
    T = labels @ labels.T  # 目标矩阵
    
    for layer_idx in range(model.config.num_hidden_layers):
        H = hidden_states[layer_idx][0].cpu().numpy()  # 取 batch 0
        # 使用采样方式计算近似 KTA
        kta_value = compute_KTA_approx(H, T, sample_size=100)
        if not np.isnan(kta_value):
            kta_results[layer_idx].append(kta_value)

# 计算每层 KTA 的均值和标准差
kta_mean = {layer: np.mean(values) for layer, values in kta_results.items() if values}
kta_std = {layer: np.std(values) for layer, values in kta_results.items() if values}

for layer in sorted(kta_mean.keys()):
    print(f"Layer {layer} KTA: {kta_mean[layer]:.4f} ± {kta_std[layer]:.4f}")

print("Final KTA Results:", kta_mean)
