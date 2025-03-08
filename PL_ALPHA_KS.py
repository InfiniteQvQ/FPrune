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

# ✅ CKA 计算修正
def compute_cka(H1, H2):
    """ 计算 CKA 相似性，防止数值溢出 """
    H1, H2 = H1.to(torch.float32), H2.to(torch.float32)  # **避免 float16 误差**
    
    # **Gram 矩阵归一化**
    K1, K2 = H1 @ H1.T, H2 @ H2.T
    K1 = K1 / (torch.norm(K1, dim=-1, keepdim=True) + 1e-6)
    K2 = K2 / (torch.norm(K2, dim=-1, keepdim=True) + 1e-6)

    # **计算 CKA**
    norm1, norm2 = torch.norm(K1), torch.norm(K2)
    similarity = torch.sum(K1 * K2) / (norm1 * norm2 + 1e-6)
    
    # **防止 NaN 传播**
    return similarity if not torch.isnan(similarity) else torch.tensor(0.0, device=device)

# ✅ 计算层间 CKA
cka_values = []
for i in range(num_layers - 1):
    H1, H2 = hidden_states[i][0], hidden_states[i + 1][0]  # 取 batch 0
    cka_values.append(compute_cka(H1, H2).item())

# **归一化 CKA**
cka_values = torch.tensor(cka_values, dtype=torch.float32).to(device)
cka_values = (cka_values - cka_values.min()) / (cka_values.max() - cka_values.min())
cka_values = torch.cat([cka_values, cka_values[-1].unsqueeze(0)])  # **补齐最后一层**

print(f"修正后的 CKA 层间相似性: {cka_values.cpu().numpy()}")
