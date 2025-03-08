import torch
import numpy as np
from transformers import LlamaModel, AutoTokenizer

# 🚀 计算 CKA 相似性
def cka_similarity(X, Y):
    X, Y = X.to(torch.float32), Y.to(torch.float32)  
    K_X = X @ X.transpose(-1, -2)  
    K_Y = Y @ Y.transpose(-1, -2)
    num = (K_X * K_Y).sum()
    denom = torch.sqrt((K_X * K_X).sum()) * torch.sqrt((K_Y * K_Y).sum())
    return num / (denom + 1e-6)

# 🚀 加载 LLaMA 7B
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

# 🚀 处理输入
text = ["LLaMA 7B Kernel Target Alignment computation."]
inputs = tokenizer(text, return_tensors="pt")
inputs.pop("token_type_ids", None)
inputs = {key: val.to(device) for key, val in inputs.items()}

# 🚀 计算隐藏状态
with torch.no_grad():
    outputs = model(**inputs)

hidden_states = outputs.hidden_states  # (num_layers, batch, seq_len, hidden_dim)

# 🚀 **确保只取 32 层 Transformer**
hidden_states = hidden_states[1:]  # **去掉第 0 层（输入嵌入层）**
num_layers = len(hidden_states)  # 🚀 现在 num_layers 应该是 32

# 🚀 计算 32 层的 CKA 相似度
cka_matrix = torch.zeros(num_layers, num_layers).to(device)

for i in range(num_layers):
    for j in range(i, num_layers):  
        cka_matrix[i, j] = cka_similarity(hidden_states[i][0], hidden_states[j][0])
        cka_matrix[j, i] = cka_matrix[i, j]  # 对称矩阵

cka_importance = cka_matrix.mean(dim=1)  # 计算每层的平均 CKA 相似性

# 🚀 ESD 剪枝比例
esd_pruning_ratios = torch.tensor([
    0.5704, 0.6176, 0.6315, 0.6307, 0.6528, 0.6482, 0.6300, 0.5921, 0.5973, 0.5680,
    0.5870, 0.5893, 0.5989, 0.6108, 0.6187, 0.6681, 0.6586, 0.7156, 0.7905, 0.7437,
    0.7946, 0.8248, 0.7700, 0.7629, 0.8121, 0.8520, 0.8312, 0.8414, 0.7869, 0.8296,
    0.8414, 0.7317
]).to(device)

# **🚀 确保 ESD 维度匹配**
if esd_pruning_ratios.shape[0] != num_layers:
    esd_pruning_ratios = esd_pruning_ratios[:num_layers]  # **确保 ESD 也是 32 层**

# **🚀 反转 CKA，确保高 CKA 重要性低**
cka_importance = 1 - cka_importance  

# 🚀 归一化 CKA 到 0~1
cka_importance = (cka_importance - cka_importance.min()) / (cka_importance.max() - cka_importance.min())

# **最终剪枝比例（让高重要性层剪枝更少）**
adjusted_pruning_ratios = esd_pruning_ratios * (1 - 0.5 * cka_importance)

# **🚀 归一化，保持剪枝比例均值不变**
original_mean = esd_pruning_ratios.mean()
adjusted_mean = adjusted_pruning_ratios.mean()
final_pruning_ratios = adjusted_pruning_ratios * (original_mean / adjusted_mean)

print("Final Adjusted Pruning Ratios:", final_pruning_ratios.cpu().numpy())
print(final_pruning_ratios.mean())
a = []
for i in final_pruning_ratios.cpu().numpy():
    a.append(i)
print(a)
