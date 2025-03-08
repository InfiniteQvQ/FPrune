import torch
import numpy as np
from transformers import LlamaModel, AutoTokenizer

# ✅ 设备设置（自动选择 GPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ 加载 LLaMA 7B 模型
model_name = "pinkmanlove/llama-7b-hf"
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceM4/llama-7b-tokenizer")
model = LlamaModel.from_pretrained(
    model_name,
    cache_dir="/root/autodl-tmp/llm_weights",
    output_hidden_states=True,
    torch_dtype=torch.float16,  
    device_map="auto"  # **自动分配多个 GPU**
)

# ✅ 输入处理
text = ["LLaMA 7B Kernel Target Alignment computation."]
inputs = tokenizer(text, return_tensors="pt")
inputs.pop("token_type_ids", None)
inputs = {key: val.to(device) for key, val in inputs.items()}

# ✅ 计算模型隐藏状态
with torch.no_grad():
    outputs = model(**inputs)

hidden_states = outputs.hidden_states  # (num_layers, batch, seq_len, hidden_dim)
num_layers = model.config.num_hidden_layers  # 获取 LLaMA 层数
print(f"模型层数: {num_layers}")

# ✅ 你已有的剪枝比例（来自 ESD）
esd_pruning_ratios = torch.tensor([
    0.5704, 0.6176, 0.6315, 0.6307, 0.6529, 0.6482, 0.6301, 0.5922, 
    0.5974, 0.5680, 0.5871, 0.5894, 0.5990, 0.6109, 0.6188, 0.6681,
    0.6587, 0.7156, 0.7906, 0.7438, 0.7946, 0.8248, 0.7701, 0.7629, 
    0.8122, 0.8521, 0.8313, 0.8415, 0.7869, 0.8297, 0.8414, 0.7317
], dtype=torch.float32).to(device)

# ✅ 计算 CKA 作为层间相似性
def compute_cka(H1, H2):
    """ 计算 CKA 相似性 """
    H1, H2 = H1.to(torch.float32), H2.to(torch.float32)
    K1, K2 = H1 @ H1.T, H2 @ H2.T  # Gram 矩阵
    norm1, norm2 = torch.norm(K1), torch.norm(K2)
    return torch.sum(K1 * K2) / (norm1 * norm2 + 1e-6)

# 计算层间 CKA
cka_values = []
for i in range(num_layers - 1):
    H1, H2 = hidden_states[i][0], hidden_states[i + 1][0]  # 取 batch 0
    cka_values.append(compute_cka(H1, H2).item())

# 归一化 CKA，使其范围在 [0,1]
cka_values = torch.tensor(cka_values, dtype=torch.float32).to(device)
cka_values = (cka_values - cka_values.min()) / (cka_values.max() - cka_values.min())
cka_values = torch.cat([cka_values, cka_values[-1].unsqueeze(0)])  # 补齐最后一层

print(f"CKA 层间相似性: {cka_values.cpu().numpy()}")

# ✅ 计算最终剪枝比例
alpha = 0.3  # 调整 CKA 影响权重
adjustment_factor = torch.exp(-alpha * cka_values)  # 指数平滑，避免大幅度调整
adjusted_pruning_ratios = esd_pruning_ratios * adjustment_factor

# ✅ 归一化，确保剪枝比例均值不变
original_mean = esd_pruning_ratios.mean()
adjusted_mean = adjusted_pruning_ratios.mean()
final_pruning_ratios = adjusted_pruning_ratios * (original_mean / adjusted_mean)

print("最终剪枝比例:", final_pruning_ratios.cpu().numpy())
