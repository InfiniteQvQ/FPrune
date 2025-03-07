import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
from transformers import LlamaModel, AutoTokenizer



def compute_KTA_approx(H, T, sample_size=1000):
    # H: (seq_len, hidden_dim)
    # T: (seq_len, seq_len) 目标矩阵，需要相应采样
    num_tokens = H.shape[0]
    sample_idx = np.random.choice(num_tokens, size=min(sample_size, num_tokens), replace=False)
    H_sample = H[sample_idx, :]
    T_sample = T[np.ix_(sample_idx, sample_idx)]
    
    H_sample_norm = np.linalg.norm(H_sample, axis=1, keepdims=True) + 1e-6
    H_sample = H_sample / H_sample_norm
    H_sample = np.clip(H_sample, -1.0, 1.0)  # 归一化和剪裁

    K_sample = H_sample @ H_sample.T
    T_sample = T_sample @ T_sample.T

    K_norm = np.sqrt(np.sum(K_sample**2)) + 1e-6
    T_norm = np.sqrt(np.sum(T_sample**2)) + 1e-6
    inner_product = np.sum(K_sample * T_sample)
    return inner_product / (K_norm * T_norm)


# **使用 `device_map="auto"` 让 HF 自动分配模型到多个 GPU**
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



model_name = "pinkmanlove/llama-7b-hf"
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceM4/llama-7b-tokenizer")

model = LlamaModel.from_pretrained(  # **确保是 `LlamaModel` 而不是 `LlamaForCausalLM`**
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

# 计算 KTA 多次平均
kta_results = {i: [] for i in range(model.config.num_hidden_layers)}  
num_runs = 10  

for _ in range(num_runs):
    with torch.no_grad():
        outputs = model(**inputs)

    hidden_states = outputs.hidden_states  
    seq_len = hidden_states[0].shape[1]  
    labels = np.random.choice([-1, 1], size=(seq_len, seq_len))  

    for layer_idx in range(model.config.num_hidden_layers):
        H = hidden_states[layer_idx][0].cpu().numpy()  
        
        # **避免 NaN 传播**
        if np.isnan(H).any():
            print(f"Warning: NaN detected in layer {layer_idx}, skipping")
            continue
        
        K = np.matmul(H, H.T)  
        kta_value = compute_KTA_approx(K, labels)
        
        # **忽略 `NaN` 结果**
        if not np.isnan(kta_value):
            kta_results[layer_idx].append(kta_value)

# 计算平均 KTA
kta_mean = {layer: np.mean(values) for layer, values in kta_results.items() if values}
kta_std = {layer: np.std(values) for layer, values in kta_results.items() if values}

for layer in kta_mean:
    print(f"Layer {layer} KTA: {kta_mean[layer]:.4f} ± {kta_std[layer]:.4f}")

print("Final KTA Results:", kta_mean)

