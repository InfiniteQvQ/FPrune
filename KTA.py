import torch
import numpy as np
from transformers import LlamaModel, AutoTokenizer


def compute_KTA(H, T):
    """
    计算 Kernel Target Alignment (KTA)，严格按照标准定义：
    KTA = <K, T> / (||K||_F * ||T||_F)
    
    其中：
    - K = H @ H^T （核矩阵）
    - T = T @ T^T （目标矩阵）
    """
    # **Step 1: 确保 H 是 float64 以避免溢出**
    H = H.astype(np.float64)
    
    # **Step 2: 归一化 H**
    H = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-6)  # 避免除 0
    H = np.clip(H, -1, 1)  # 限制值域

    # **Step 3: 计算核矩阵 K**
    K = np.matmul(H, H.T)

    # **Step 4: 计算目标矩阵 T**
    T = np.matmul(T, T.T)

    # **Step 5: 计算 Frobenius 范数**
    K_norm = np.sqrt(np.sum(K**2)) + 1e-6
    T_norm = np.sqrt(np.sum(T**2)) + 1e-6

    # **Step 6: 计算 KTA**
    inner_product = np.sum(K * T)
    return inner_product / (K_norm * T_norm)


# **加载 Llama 模型**
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "pinkmanlove/llama-7b-hf"

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceM4/llama-7b-tokenizer")
model = LlamaModel.from_pretrained(
    model_name,
    cache_dir="/root/autodl-tmp/llm_weights",
    output_hidden_states=True,
    torch_dtype=torch.float16,  # 仍然使用 float16 进行推理
    device_map="auto"
)

# **输入处理**
text = ["LLaMA 7B Kernel Target Alignment computation."]
inputs = tokenizer(text, return_tensors="pt")
inputs.pop("token_type_ids", None)
inputs = {key: val.to(device) for key, val in inputs.items()}

num_runs = 10  # 计算 10 次，取平均值
kta_results = {i: [] for i in range(model.config.num_hidden_layers)}

for run in range(num_runs):
    with torch.no_grad():
        outputs = model(**inputs)

    hidden_states = outputs.hidden_states  # (num_layers, batch, seq_len, hidden_dim)
    seq_len = hidden_states[0].shape[1]

    # **生成 T**
    labels = np.random.choice([-1, 1], size=(seq_len, 1))
    T = labels @ labels.T  # 目标矩阵

    for layer_idx in range(model.config.num_hidden_layers):
        H = hidden_states[layer_idx][0].cpu().numpy()  # 取 batch 0
        kta_value = compute_KTA(H, T)
        if not np.isnan(kta_value):
            kta_results[layer_idx].append(kta_value)

# **计算均值和标准差**
kta_mean = {layer: np.mean(values) for layer, values in kta_results.items() if values}
kta_std = {layer: np.std(values) for layer, values in kta_results.items() if values}

for layer in sorted(kta_mean.keys()):
    print(f"Layer {layer} KTA: {kta_mean[layer]:.4f} ± {kta_std[layer]:.4f}")

print("Final KTA Results:", kta_mean)
