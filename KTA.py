import torch
import numpy as np
from transformers import LlamaModel, AutoTokenizer

def compute_KTA_approx(H, T, sample_size=512):
    """
    改进版 KTA 计算：
    - 使用 SVD 近似 \( K \) ，减少数值溢出
    - 使用稳定目标矩阵 \( T \)，避免随机性影响
    - 进行归一化，减少极端值影响
    """
    num_tokens = H.shape[0]
    sample_idx = np.random.choice(num_tokens, size=min(sample_size, num_tokens), replace=False)
    H_sample = H[sample_idx, :]
    T_sample = T[np.ix_(sample_idx, sample_idx)]

    # 使用 float64 提高精度
    H_sample = H_sample.astype(np.float64)

    # **使用 SVD 进行核矩阵近似，减少计算量**
    U, S, Vt = np.linalg.svd(H_sample, full_matrices=False)
    H_sample = U @ np.diag(S)  # 近似降维后的 H

    # 归一化 H_sample
    norm_H = np.linalg.norm(H_sample, axis=1, keepdims=True) + 1e-6
    H_sample = H_sample / norm_H
    H_sample = np.clip(H_sample, -1, 1)

    # 计算核矩阵 K_sample
    K_sample = H_sample @ H_sample.T

    # **使用固定的正交目标矩阵 T，避免随机性影响**
    T_sample = np.eye(sample_size)  # 直接使用单位矩阵

    # 计算 Frobenius 范数（加 epsilon 防止 0 除法）
    K_norm = np.sqrt(np.sum(K_sample**2)) + 1e-6
    T_norm = np.sqrt(np.sum(T_sample**2)) + 1e-6
    inner_product = np.sum(K_sample * T_sample)

    return inner_product / (K_norm * T_norm)


# **加载模型**
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "pinkmanlove/llama-7b-hf"

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceM4/llama-7b-tokenizer")
model = LlamaModel.from_pretrained(
    model_name,
    cache_dir="/root/autodl-tmp/llm_weights",
    output_hidden_states=True,
    torch_dtype=torch.float16,
    device_map="auto"
)

# 处理输入
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
    
    # **使用固定的单位矩阵 T**
    T = np.eye(seq_len)  

    for layer_idx in range(model.config.num_hidden_layers):
        H = hidden_states[layer_idx][0].cpu().numpy()  # 取 batch 0
        # 使用改进的 KTA 计算
        kta_value = compute_KTA_approx(H, T, sample_size=512)
        if not np.isnan(kta_value):
            kta_results[layer_idx].append(kta_value)

# 计算每层 KTA 的均值和标准差
kta_mean = {layer: np.mean(values) for layer, values in kta_results.items() if values}
kta_std = {layer: np.std(values) for layer, values in kta_results.items() if values}

for layer in sorted(kta_mean.keys()):
    print(f"Layer {layer} KTA: {kta_mean[layer]:.4f} ± {kta_std[layer]:.4f}")

print("Final KTA Results:", kta_mean)
