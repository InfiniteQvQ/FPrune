import torch
import numpy as np
from transformers import LlamaModel, AutoTokenizer

def compute_KTA_approx(H, sample_size=512):
    """
    近似计算 KTA，减少数值溢出并保证计算稳定性
    """
    num_tokens = H.shape[0]  # 真实 token 长度
    sample_size = min(sample_size, num_tokens)

    # 采样
    sample_idx = np.random.choice(num_tokens, size=sample_size, replace=False)
    H_sample = H[sample_idx, :]
    H_sample = H_sample.astype(np.float32)
    # SVD 降维
    U, S, Vt = np.linalg.svd(H_sample, full_matrices=False)
    rank = min(sample_size, H_sample.shape[1])
    H_sample = U[:, :rank] @ np.diag(S[:rank]) @ Vt[:rank, :]

    # 归一化
    H_sample = H_sample / (np.linalg.norm(H_sample, axis=1, keepdims=True) + 1e-6)
    H_sample = np.clip(H_sample, -1, 1)

    # 计算核矩阵
    K_sample = np.matmul(H_sample, H_sample.T)
    K_sample /= np.max(np.abs(K_sample)) + 1e-6

    # 计算目标矩阵 T
    T_sample = np.matmul(H_sample, H_sample.T)
    T_sample /= np.max(np.abs(T_sample)) + 1e-6

    # 计算 KTA
    inner_product = np.trace(K_sample @ T_sample)
    K_norm = np.sqrt(np.sum(K_sample**2)) + 1e-6
    T_norm = np.sqrt(np.sum(T_sample**2)) + 1e-6

    return inner_product / (K_norm * T_norm)


# **加载 Llama 模型**
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

    for layer_idx in range(model.config.num_hidden_layers):
        H = hidden_states[layer_idx][0].cpu().numpy()  # 取 batch 0
        kta_value = compute_KTA_approx(H, sample_size=256)  # 降低 sample_size
        if not np.isnan(kta_value):
            kta_results[layer_idx].append(kta_value)

# **计算均值和标准差**
kta_mean = {layer: np.mean(values) for layer, values in kta_results.items() if values}
kta_std = {layer: np.std(values) for layer, values in kta_results.items() if values}

for layer in sorted(kta_mean.keys()):
    print(f"Layer {layer} KTA: {kta_mean[layer]:.4f} ± {kta_std[layer]:.4f}")

print("Final KTA Results:", kta_mean)
