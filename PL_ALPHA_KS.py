import torch
import numpy as np

# ✅ 计算 CKA 相似性（加入 NaN 处理）
def compute_cka(H1, H2):
    """
    计算 CKA（Centered Kernel Alignment），衡量两个层之间的相似性
    H1, H2: 形状为 (seq_len, hidden_dim) 的张量
    """
    eps = 1e-6  # 避免 0 除法
    H1 = H1.float() - H1.mean(dim=0, keepdim=True)  # 中心化
    H2 = H2.float() - H2.mean(dim=0, keepdim=True)  # 中心化

    K1 = H1 @ H1.T  # Gram 矩阵
    K2 = H2 @ H2.T  # Gram 矩阵

    norm1 = torch.norm(K1, p='fro') + eps  # Frobenius 范数
    norm2 = torch.norm(K2, p='fro') + eps  # Frobenius 范数

    cka_value = torch.sum(K1 * K2) / (norm1 * norm2)

    if torch.isnan(cka_value):
        return 0.0  # 处理 NaN，防止错误传播
    return cka_value.item()

# ✅ 计算所有层的 CKA 相似性
def compute_cka_all_layers(hidden_states):
    """
    计算所有相邻层的 CKA
    """
    num_layers = len(hidden_states)
    cka_values = []

    for i in range(num_layers - 1):  # 计算相邻层的相似性
        H1 = hidden_states[i].squeeze(0)  # (seq_len, hidden_dim)
        H2 = hidden_states[i + 1].squeeze(0)  # (seq_len, hidden_dim)

        if H1.numel() == 0 or H2.numel() == 0:  # 避免空矩阵计算
            cka_values.append(0.0)
            continue

        cka_score = compute_cka(H1, H2)
        cka_values.append(cka_score)

    return np.array(cka_values)

# ✅ 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "pinkmanlove/llama-7b-hf"

from transformers import LlamaModel, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceM4/llama-7b-tokenizer")

model = LlamaModel.from_pretrained(
    model_name,
    cache_dir="/root/autodl-tmp/llm_weights",
    output_hidden_states=True,
    torch_dtype=torch.float16,
    device_map="auto"  # **自动分配多个 GPU**
)

# ✅ 处理输入
text = ["LLaMA 7B Kernel Target Alignment computation."]
inputs = tokenizer(text, return_tensors="pt")
inputs.pop("token_type_ids", None)
inputs = {key: val.to(device) for key, val in inputs.items()}

# ✅ 前向传播，获取隐藏层
with torch.no_grad():
    outputs = model(**inputs)

hidden_states = outputs.hidden_states  # List[num_layers, batch_size, seq_len, hidden_dim]

# ✅ 计算 CKA
cka_values = compute_cka_all_layers(hidden_states)

# ✅ 过滤 `NaN` 并修正数值范围
cka_values = np.nan_to_num(cka_values, nan=0.0)  # 替换 NaN
cka_values = (cka_values - np.min(cka_values)) / (np.max(cka_values) - np.min(cka_values) + 1e-6)  # 归一化

# ✅ ESD 计算得到的剪枝比例
esd_pruning_ratios = np.array([
    0.570, 0.617, 0.631, 0.630, 0.652, 0.648, 0.630, 0.592, 0.597, 0.568,
    0.587, 0.589, 0.598, 0.610, 0.618, 0.668, 0.658, 0.715, 0.790, 0.743,
    0.794, 0.824, 0.770, 0.762, 0.812, 0.852, 0.831, 0.841, 0.786, 0.829,
    0.841, 0.731
])

# ✅ CKA 修正 ESD 剪枝比例
alpha = 0.5  # CKA 影响力（可以调整）
adjusted_pruning_ratios = esd_pruning_ratios * (1 - alpha * cka_values)

# ✅ 重新归一化，使得剪枝比例总和保持不变
scaling_factor = np.mean(esd_pruning_ratios) / np.mean(adjusted_pruning_ratios)
adjusted_pruning_ratios *= scaling_factor

# ✅ 输出结果
print("✅ 原始 ESD 剪枝比例:", esd_pruning_ratios)
print("✅ 计算出的 CKA 相似度:", cka_values)
print("✅ CKA 修正后剪枝比例:", adjusted_pruning_ratios)
