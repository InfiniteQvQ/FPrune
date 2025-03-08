import torch
import numpy as np
from transformers import LlamaModel, AutoTokenizer

# **🚀 计算 CKA 相似度**
def cka_similarity(X, Y):
    """ 计算 CKA 相似度，衡量 Transformer 层之间的相似性。 """
    X, Y = X.to(torch.float32), Y.to(torch.float32)  # 计算稳定性
    K_X = X @ X.transpose(-1, -2)  # Gram 矩阵
    K_Y = Y @ Y.transpose(-1, -2)

    num = (K_X * K_Y).sum()
    denom = torch.sqrt((K_X * K_X).sum()) * torch.sqrt((K_Y * K_Y).sum())

    return num / (denom + 1e-6)  # 避免除 0

# **🚀 加载 LLaMA 7B**
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "pinkmanlove/llama-7b-hf"

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceM4/llama-7b-tokenizer")
model = LlamaModel.from_pretrained(
    model_name,
    cache_dir="/root/autodl-tmp/llm_weights",
    output_hidden_states=True,
    torch_dtype=torch.float16,
    device_map="auto"  # **自动分配多个 GPU**
)

# **🚀 输入文本**
text = ["LLaMA 7B Kernel Target Alignment computation."]
inputs = tokenizer(text, return_tensors="pt")
inputs.pop("token_type_ids", None)
inputs = {key: val.to(device) for key, val in inputs.items()}

# **🚀 获取 32 层隐藏状态**
with torch.no_grad():
    outputs = model(**inputs)

hidden_states = outputs.hidden_states  # (num_layers, batch, seq_len, hidden_dim)
num_layers = len(hidden_states)
seq_len = hidden_states[0].shape[1]

# **🚀 计算 32 层的 CKA 相似度**
cka_matrix = torch.zeros(num_layers, num_layers).to(device)

for i in range(num_layers):
    for j in range(i, num_layers):  # 只计算上三角
        cka_matrix[i, j] = cka_similarity(hidden_states[i][0], hidden_states[j][0])
        cka_matrix[j, i] = cka_matrix[i, j]  # 对称矩阵

# **🚀 计算每层 CKA 重要性（与其他层的平均相似度）**
cka_importance = cka_matrix.mean(dim=1)

# **🚀 ESD 剪枝比例**
esd_pruning_ratios = torch.tensor([
    0.5704, 0.6176, 0.6315, 0.6307, 0.6528, 0.6482, 0.6300, 0.5921, 0.5973, 0.5680,
    0.5870, 0.5893, 0.5989, 0.6108, 0.6187, 0.6681, 0.6586, 0.7156, 0.7905, 0.7437,
    0.7946, 0.8248, 0.7700, 0.7629, 0.8121, 0.8520, 0.8312, 0.8414, 0.7869, 0.8296,
    0.8414, 0.7317
]).to(device)

# **🚀 结合 CKA 重新调整剪枝比例**
min_ratio, max_ratio = 0.3, 0.9  # 剪枝比例范围

# 归一化 CKA 重要性
cka_importance = (cka_importance - cka_importance.min()) / (cka_importance.max() - cka_importance.min())

# **计算最终剪枝比例**
adjusted_pruning_ratios = min_ratio + (max_ratio - min_ratio) * (1 - cka_importance) * esd_pruning_ratios

# **🚀 归一化，确保整体 sparsity_ratio 不变**
scaler = esd_pruning_ratios.sum() / adjusted_pruning_ratios.sum()
final_pruning_ratios = adjusted_pruning_ratios * scaler

print("Final Adjusted Pruning Ratios:", final_pruning_ratios.cpu().numpy())
