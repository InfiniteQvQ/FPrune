import torch
import numpy as np
from transformers import LlamaModel, AutoTokenizer

def compute_PL_Alpha_KS(H, T, activation="tanh"):
    """
    计算 PL_Alpha_KS，每层计算 K 和 T 之间的对齐程度。
    
    Args:
        H (numpy.ndarray): 形状 (seq_len, hidden_dim) 的隐藏状态
        T (numpy.ndarray): 形状 (seq_len, seq_len) 的目标矩阵
        activation (str): "tanh" 或 "relu" 激活函数
    
    Returns:
        float: 计算得到的 PL_Alpha_KS 值
    """
    # 归一化 H
    H = H.astype(np.float64)
    norm_H = np.linalg.norm(H, axis=1, keepdims=True) + 1e-6
    H = H / norm_H
    
    # 计算核矩阵 K
    K = H @ H.T  # 线性核
    if activation == "tanh":
        K = np.tanh(K)  # 非线性变换
    elif activation == "relu":
        K = np.maximum(K, 0)  # ReLU 变换
    else:
        raise ValueError("Unsupported activation function")
    
    # 归一化 T
    T_norm = np.linalg.norm(T, axis=1, keepdims=True) + 1e-6
    T = T / T_norm
    
    # 计算 PL_Alpha_KS
    K_norm = np.sqrt(np.sum(K**2)) + 1e-6
    T_norm = np.sqrt(np.sum(T**2)) + 1e-6
    inner_product = np.sum(K * T)
    
    return inner_product / (K_norm * T_norm)


# **加载 LLaMA 模型**
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
text = ["LLaMA 7B PL_Alpha_KS computation."]
inputs = tokenizer(text, return_tensors="pt")
inputs.pop("token_type_ids", None)
inputs = {key: val.to(device) for key, val in inputs.items()}

num_runs = 10  # 取 10 次平均
pl_alpha_ks_results = {i: [] for i in range(model.config.num_hidden_layers)}

for run in range(num_runs):
    with torch.no_grad():
        outputs = model(**inputs)

    hidden_states = outputs.hidden_states  # tuple of (num_layers, batch, seq_len, hidden_dim)
    seq_len = hidden_states[0].shape[1]

    # **目标矩阵 T（Identity 矩阵）**
    T = np.eye(seq_len).astype(np.float32)

    for layer_idx in range(model.config.num_hidden_layers):
        H = hidden_states[layer_idx][0].cpu().numpy()  # 取 batch 0
        pl_alpha_ks_value = compute_PL_Alpha_KS(H, T, activation="tanh")
        if not np.isnan(pl_alpha_ks_value):
            pl_alpha_ks_results[layer_idx].append(pl_alpha_ks_value)

# **计算均值和标准差**
pl_alpha_ks_mean = {layer: np.mean(values) for layer, values in pl_alpha_ks_results.items() if values}
pl_alpha_ks_std = {layer: np.std(values) for layer, values in pl_alpha_ks_results.items() if values}

for layer in sorted(pl_alpha_ks_mean.keys()):
    print(f"Layer {layer} PL_Alpha_KS: {pl_alpha_ks_mean[layer]:.4f} ± {pl_alpha_ks_std[layer]:.4f}")

print("Final PL_Alpha_KS Results:", pl_alpha_ks_mean)

res = []
for i, j in pl_alpha_ks_mean.items():
    res.append(j)

print(res)
