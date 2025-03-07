import torch
import numpy as np
from transformers import LlamaModel, AutoTokenizer

def compute_KTA_approx(H, T, sample_size=512):
    """
    近似计算 KTA，避免数值溢出，并保证 T 和 K 形状匹配
    """
    num_tokens = H.shape[0]  # 实际 token 长度
    sample_size = min(sample_size, num_tokens)  # 不能超过实际 token 长度

    # 采样索引
    sample_idx = np.random.choice(num_tokens, size=sample_size, replace=False)
    H_sample = H[sample_idx, :]
    T_sample = T[np.ix_(sample_idx, sample_idx)]  # 保证形状匹配

    # **修正：转换为 float32，避免 NumPy SVD 不支持 float16**
    H_sample = H_sample.astype(np.float32)

    # **SVD 降维减少数值溢出**
    U, S, Vt = np.linalg.svd(H_sample, full_matrices=False)
    H_sample = U @ np.diag(S)

    # **归一化 H_sample**
    norm_H = np.linalg.norm(H_sample, axis=1, keepdims=True) + 1e-6
    H_sample = H_sample / norm_H
    H_sample = np.clip(H_sample, -1, 1)

    # 计算核矩阵 K_sample
    K_sample = H_sample @ H_sample.T

    # **T_sample 归一化，防止 NaN**
    T_sample = T_sample.astype(np.float32)
    norm_T = np.linalg.norm(T_sample, axis=1, keepdims=True) + 1e-6
    T_sample = T_sample / norm_T

    # **计算 KTA**
    K_norm = np.sqrt(np.sum(K_sample**2)) + 1e-6
    T_norm = np.sqrt(np.sum(T_sample**2)) + 1e-6
    inner_product = np.sum(K_sample * T_sample)

    return inner_product / (K_norm * T_norm)


# **加载 Llama 模型**
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "pinkmanlove/llama-7b-hf"

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceM4/llama-7b-tokenizer")
model = LlamaModel.from_pretrained(
    model_name,
    cache_dir="/root/autodl-tmp/llm_weights",
    output_hidden_states=True,
    torch_dtype=torch.float16,  # **仍然使用 float16 进行推理**
    device_map="auto"
)

# **输入处理**
text = ["LLaMA 7B Kernel Target Alignment computation."]
inputs = tokenizer(text, return_tensors="pt")
inputs.pop("token_type_ids", None)
inputs = {key: val.to(device) for key, val in inputs.items()}

num_runs = 10  # 取 10 次平均
kta_results = {i: [] for i in range(model.config.num_hidden_layers)}

for run in range(num_runs):
    with torch.no_grad():
        outputs = model(**inputs)

    hidden_states = outputs.hidden_states  # tuple of (num_layers, batch, seq_len, hidden_dim)
    seq_len = hidden_states[0].shape[1]

    # **保证 T 形状匹配**
    T = np.eye(seq_len).astype(np.float32)  # **转换为 float32，防止计算误差**

    for layer_idx in range(model.config.num_hidden_layers):
        H = hidden_states[layer_idx][0].cpu().numpy()  # 取 batch 0
        kta_value = compute_KTA_approx(H, T, sample_size=512)
        if not np.isnan(kta_value):
            kta_results[layer_idx].append(kta_value)

# **计算均值和标准差**
kta_mean = {layer: np.mean(values) for layer, values in kta_results.items() if values}
kta_std = {layer: np.std(values) for layer, values in kta_results.items() if values}

for layer in sorted(kta_mean.keys()):
    print(f"Layer {layer} KTA: {kta_mean[layer]:.4f} ± {kta_std[layer]:.4f}")

print("Final KTA Results:", kta_mean)

res = []
for i, j in kta_mean.items():
    res.append(j)

print(res)