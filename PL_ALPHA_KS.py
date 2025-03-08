import torch
import numpy as np
from transformers import LlamaModel, AutoTokenizer

### 计算 CKA 相似度 ###
def compute_cka(H1, H2):
    H1, H2 = H1.cpu().float(), H2.cpu().float()
    K1, K2 = H1 @ H1.T, H2 @ H2.T  # 计算 Gram 矩阵
    K1_norm, K2_norm = torch.norm(K1, 'fro'), torch.norm(K2, 'fro')
    return (torch.sum(K1 * K2) / (K1_norm * K2_norm)).item()

### 计算 ESD + CKA 修正的剪枝比例 ###
def adjust_esd_with_cka(esd_ratios, cka_values, min_ratio=0.5, max_ratio=0.9):
    cka_importance = 1 - np.array(cka_values)  # CKA 低的层更重要
    cka_importance = (cka_importance - cka_importance.min()) / (cka_importance.max() - cka_importance.min())  # 归一化

    adjusted_pruning_ratios = min_ratio + (max_ratio - min_ratio) * (1 - cka_importance) * esd_ratios

    # 保持原始 ESD 的均值不变（避免影响模型整体 sparsity）
    original_mean = np.mean(esd_ratios)
    adjusted_mean = np.mean(adjusted_pruning_ratios)
    scale_factor = original_mean / adjusted_mean
    adjusted_pruning_ratios *= scale_factor  # 归一化调整

    return adjusted_pruning_ratios

### 加载模型 ###
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

### 输入处理 ###
text = ["LLaMA 7B Kernel Target Alignment computation."]
inputs = tokenizer(text, return_tensors="pt")
inputs.pop("token_type_ids", None)
inputs = {key: val.to(device) for key, val in inputs.items()}

### 计算隐藏状态 ###
with torch.no_grad():
    outputs = model(**inputs)
hidden_states = outputs.hidden_states  # tuple of (num_layers, batch, seq_len, hidden_dim)

num_layers = len(hidden_states) - 1  # 计算相邻层 CKA
cka_values = []

for i in range(num_layers):
    H1 = hidden_states[i].squeeze(0)
    H2 = hidden_states[i + 1].squeeze(0)
    cka_values.append(compute_cka(H1, H2))

### 你的 ESD 剪枝比例 ###
esd_ratios = np.array([
    0.57, 0.617, 0.631, 0.63, 0.652, 0.648, 0.63, 0.592, 0.597, 0.568, 
    0.587, 0.589, 0.598, 0.61, 0.618, 0.668, 0.658, 0.715, 0.79, 0.743, 
    0.794, 0.824, 0.77, 0.762, 0.812, 0.852, 0.831, 0.841, 0.786, 0.829, 
    0.841, 0.731
])

### 计算修正后的剪枝比例 ###
adjusted_pruning_ratios = adjust_esd_with_cka(esd_ratios, cka_values)

### 打印结果 ###
print(f"✅ 原始 ESD 剪枝比例: {esd_ratios}")
print(f"✅ 计算出的 CKA 相似度: {cka_values}")
print(f"✅ CKA 修正后剪枝比例: {adjusted_pruning_ratios}")
