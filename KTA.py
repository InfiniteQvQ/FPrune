import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
from transformers import LlamaModel, AutoTokenizer



def compute_KTA(H, T):
    """ 计算 Kernel Target Alignment (KTA) """

    H = H.astype(np.float64)  # **使用 float64 提高精度**
    
    # **归一化 H，防止溢出**
    H_norm = np.linalg.norm(H, axis=1, keepdims=True) + 1e-6  # **确保不为 0**
    H = H / H_norm
    H = np.clip(H, -1, 1)  # **限制数值范围，防止极端值影响计算**

    # **log-scaling 防止爆炸**
    H = np.sign(H) * np.log1p(np.abs(H))  

    # **计算核矩阵 K 和目标矩阵 T**
    K = np.matmul(H, H.T)  
    T = np.matmul(T, T.T)  

    # **计算 Frobenius 范数**
    K_norm = np.sqrt(np.sum(K**2)) + 1e-6
    T_norm = np.sqrt(np.sum(T**2)) + 1e-6

    inner_product = np.sum(K * T)
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
        kta_value = compute_KTA(K, labels)
        
        # **忽略 `NaN` 结果**
        if not np.isnan(kta_value):
            kta_results[layer_idx].append(kta_value)

# 计算平均 KTA
kta_mean = {layer: np.mean(values) for layer, values in kta_results.items() if values}
kta_std = {layer: np.std(values) for layer, values in kta_results.items() if values}

for layer in kta_mean:
    print(f"Layer {layer} KTA: {kta_mean[layer]:.4f} ± {kta_std[layer]:.4f}")

print("Final KTA Results:", kta_mean)

