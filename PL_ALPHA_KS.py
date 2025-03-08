import torch
import numpy as np
from scipy.special import expit  # ✅ Sigmoid

# ✅ 计算 CKA 相似性
def compute_cka(H1, H2):
    eps = 1e-6  # 避免 0 除法
    H1 = H1.float() - H1.mean(dim=0, keepdim=True)  # 中心化
    H2 = H2.float() - H2.mean(dim=0, keepdim=True)  # 中心化

    K1 = H1 @ H1.T  # Gram 矩阵
    K2 = H2 @ H2.T  # Gram 矩阵

    norm1 = torch.norm(K1, p='fro') + eps
    norm2 = torch.norm(K2, p='fro') + eps

    cka_value = torch.sum(K1 * K2) / (norm1 * norm2)
    return cka_value.item() if not torch.isnan(cka_value) else 0.0

# ✅ 计算所有层的 CKA
def compute_cka_all_layers(hidden_states):
    num_layers = len(hidden_states)
    cka_values = []

    for i in range(num_layers - 1):
        H1 = hidden_states[i].squeeze(0)
        H2 = hidden_states[i + 1].squeeze(0)

        if H1.numel() == 0 or H2.numel() == 0:
            cka_values.append(0.0)
            continue

        cka_score = compute_cka(H1, H2)
        cka_values.append(cka_score)

    return np.array(cka_values)

# ✅ 加载 Llama 模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "pinkmanlove/llama-7b-hf"
from transformers import LlamaModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceM4/llama-7b-tokenizer")
model = LlamaModel.from_pretrained(
    model_name,
    cache_dir="/root/autodl-tmp/llm_weights",
    output_hidden_states=True,
    torch_dtype=torch.float16,
    device_map="auto"
)

# ✅ 处理输入
text = ["LLaMA 7B Kernel Target Alignment computation."]
inputs = tokenizer(text, return_tensors="pt")
inputs.pop("token_type_ids", None)
inputs = {key: val.to(device) for key, val in inputs.items()}

# ✅ 计算隐藏层
with torch.no_grad():
    outputs = model(**inputs)

hidden_states = outputs.hidden_states
cka_values = compute_cka_all_layers(hidden_states)

# ✅ 归一化 CKA
cka_values = np.nan_to_num(cka_values, nan=0.0)  # 替换 NaN
cka_values = (cka_values - np.min(cka_values)) / (np.max(cka_values) - np.min(cka_values) + 1e-6)

# ✅ 计算平滑 CKA（防止极端值）
smooth_cka = np.zeros_like(cka_values)
for i in range(1, len(cka_values) - 1):
    smooth_cka[i] = (cka_values[i-1] + cka_values[i] + cka_values[i+1]) / 3
smooth_cka[0] = cka_values[0]
smooth_cka[-1] = cka_values[-1]

# ✅ 使用 Sigmoid 调整
beta = 5  # 控制 CKA 对剪枝的影响
adjusted_cka = expit(beta * (smooth_cka - 0.5))  # Sigmoid 调整

# ✅ ESD 剪枝比例
esd_pruning_ratios = np.array([
    0.570, 0.617, 0.631, 0.630, 0.652, 0.648, 0.630, 0.592, 0.597, 0.568,
    0.587, 0.589, 0.598, 0.610, 0.618, 0.668, 0.658, 0.715, 0.790, 0.743,
    0.794, 0.824, 0.770, 0.762, 0.812, 0.852, 0.831, 0.841, 0.786, 0.829,
    0.841, 0.731
])

# ✅ 结合 CKA 影响 ESD
alpha = 0.5  # CKA 影响力（可以调整）
adjusted_pruning_ratios = esd_pruning_ratios * (1 - alpha * adjusted_cka)

# ✅ 重新归一化，使剪枝总和不变
scaling_factor = np.mean(esd_pruning_ratios) / np.mean(adjusted_pruning_ratios)
adjusted_pruning_ratios *= scaling_factor

# ✅ 输出最终剪枝比例
print("✅ 原始 ESD 剪枝比例:", esd_pruning_ratios)
print("✅ 平滑后 CKA 相似度:", smooth_cka)
print("✅ Sigmoid 归一化 CKA:", adjusted_cka)
print("✅ CKA 修正后剪枝比例:", adjusted_pruning_ratios)
