import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer


def compute_KTA(H, T):
    """ 计算 Kernel Target Alignment (KTA) """
    H = H.astype(np.float32)  # **确保 H 为 float32，避免 float16 精度问题**
    
    # 归一化 H，防止溢出
    H_norm = np.linalg.norm(H, axis=1, keepdims=True) + 1e-8
    H = H / H_norm
    H = np.clip(H, -1e6, 1e6)  # 限制 H 的值域，防止溢出

    # 计算核矩阵 K 和目标矩阵 T
    K = H @ H.T  
    T = T @ T.T  

    # 计算 Frobenius 范数
    K_norm = np.sqrt(np.sum(K**2)) + 1e-8
    T_norm = np.sqrt(np.sum(T**2)) + 1e-8

    inner_product = np.sum(K * T)
    return inner_product / (K_norm * T_norm)


# **使用 `device_map="auto"` 让 HF 自动分配模型到多个 GPU**
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "pinkmanlove/llama-7b-hf"
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceM4/llama-7b-tokenizer")

model = AutoModel.from_pretrained(
    model_name,
    cache_dir="/root/autodl-tmp/llm_weights",
    output_hidden_states=True,
    torch_dtype=torch.float16,
    device_map="auto"  # **让 HF 自动分配 GPU**
)

# 处理输入
text = ["LLaMA 7B Kernel Target Alignment computation."]
inputs = tokenizer(text, return_tensors="pt")
inputs.pop("token_type_ids", None)
inputs = {key: val.to(device) for key, val in inputs.items()}  

# 计算 KTA 多次平均
num_runs = 10  
kta_results = {i: [] for i in range(model.config.num_hidden_layers)}  

for _ in range(num_runs):
    with torch.no_grad():
        outputs = model(**inputs)

    hidden_states = outputs.hidden_states  
    seq_len = hidden_states[0].shape[1]  
    labels = np.random.choice([-1, 1], size=(seq_len, seq_len))  

    for layer_idx in range(model.config.num_hidden_layers):
        H = hidden_states[layer_idx][0].cpu().numpy()  
        K = H @ H.T  
        kta_value = compute_KTA(K, labels)
        kta_results[layer_idx].append(kta_value)

# 计算平均 KTA
kta_mean = {layer: np.mean(values) for layer, values in kta_results.items()}
kta_std = {layer: np.std(values) for layer, values in kta_results.items()}

for layer in kta_mean:
    print(f"Layer {layer} KTA: {kta_mean[layer]:.4f} ± {kta_std[layer]:.4f}")

print("Final KTA Results:", kta_mean)
