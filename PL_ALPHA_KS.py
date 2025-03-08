import torch
import torch.nn.functional as F
import numpy as np
from transformers import LlamaModel, AutoTokenizer

# **🔥 确保模型用多个 GPU**
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "pinkmanlove/llama-7b-hf"

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceM4/llama-7b-tokenizer")
model = LlamaModel.from_pretrained(
    model_name,
    cache_dir="/root/autodl-tmp/llm_weights",
    output_hidden_states=True,
    torch_dtype=torch.float16,  
    device_map="auto"  # **自动分配多个 GPU**
)

# **输入处理**
text = ["LLaMA 7B Kernel Target Alignment computation."]
inputs = tokenizer(text, return_tensors="pt")
inputs.pop("token_type_ids", None)
inputs = {key: val.to(device) for key, val in inputs.items()}

# **前向传播**
with torch.no_grad():
    outputs = model(**inputs)

hidden_states = outputs.hidden_states  # tuple of (num_layers, batch, seq_len, hidden_dim)
num_layers = len(hidden_states) - 1  # 不包括 embedding 层

# **🔥 Mini-Batch Wasserstein 距离计算**
def wasserstein_distance_torch(H1, H2, eps=1e-3, max_iter=50, batch_size=1024):
    """
    计算 Sinkhorn-Wasserstein 距离，支持 mini-batch 计算，减少显存占用
    """
    H1 = H1.view(-1, 1).float()
    H2 = H2.view(-1, 1).float()

    n = H1.size(0)
    m = H2.size(0)

    # 🔥 **分批计算 Cost 矩阵**
    cost_matrix = torch.zeros(n, m, device=H1.device)
    
    for i in range(0, n, batch_size):
        for j in range(0, m, batch_size):
            sub_H1 = H1[i : i + batch_size]
            sub_H2 = H2[j : j + batch_size]
            cost_matrix[i : i + batch_size, j : j + batch_size] = torch.cdist(sub_H1, sub_H2, p=2).pow(2)

    # **初始化分布**
    a = torch.ones(n, device=H1.device) / n
    b = torch.ones(m, device=H2.device) / m

    u = torch.zeros_like(a)
    v = torch.zeros_like(b)

    # **🔥 Sinkhorn-Knopp 迭代**
    for _ in range(max_iter):
        u = -torch.logsumexp((-cost_matrix + v[None, :]) / eps, dim=1) + torch.log(a)
        v = -torch.logsumexp((-cost_matrix + u[:, None]) / eps, dim=0) + torch.log(b)

    return (u[:, None] + v[None, :] - cost_matrix).exp().sum()


# **🔥 Mini-Batch 互信息计算**
def mutual_information_torch(H1, H2, num_bins=256, batch_size=1024):
    """
    计算互信息，支持 mini-batch 计算，减少显存占用
    """
    H1 = H1.view(-1).float()
    H2 = H2.view(-1).float()

    # 🔥 **按 batch 计算直方图**
    hist2d = torch.zeros(num_bins, num_bins, device=H1.device)

    for i in range(0, H1.numel(), batch_size):
        sub_H1 = H1[i : i + batch_size]
        sub_H2 = H2[i : i + batch_size]
        hist = torch.histogram2d(sub_H1, sub_H2, bins=num_bins, weight=None, range=None)
        hist2d += hist.hist

    # 归一化
    P_xy = hist2d / torch.sum(hist2d)
    P_x = torch.sum(P_xy, dim=1, keepdim=True)
    P_y = torch.sum(P_xy, dim=0, keepdim=True)

    # 计算互信息
    P_xy_nonzero = P_xy[P_xy > 0]
    P_x_nonzero = P_x[P_x > 0]
    P_y_nonzero = P_y[P_y > 0]

    I_xy = (P_xy_nonzero * torch.log(P_xy_nonzero / (P_x_nonzero * P_y_nonzero))).sum()
    return I_xy


# **🔥 计算 Wasserstein 和 互信息**
layerwise_wasserstein = []
layerwise_mutual_info = []

for layer_idx in range(num_layers - 1):
    H = hidden_states[layer_idx][0].to(torch.float32)
    H_next = hidden_states[layer_idx + 1][0].to(torch.float32)

    # 🔥 **使用 Mini-Batch 计算 Wasserstein**
    wd = wasserstein_distance_torch(H, H_next, batch_size=512)  # **小批量计算**
    layerwise_wasserstein.append(wd.item())

    # 🔥 **使用 Mini-Batch 计算互信息**
    mi = mutual_information_torch(H, H_next, batch_size=512)
    layerwise_mutual_info.append(mi.item())

# **归一化 Wasserstein & 互信息**
wasserstein_scores = torch.tensor(layerwise_wasserstein, device="cuda")
mutual_info_scores = torch.tensor(layerwise_mutual_info, device="cuda")
wasserstein_scores = (wasserstein_scores - wasserstein_scores.min()) / (wasserstein_scores.max() - wasserstein_scores.min())
mutual_info_scores = (mutual_info_scores - mutual_info_scores.min()) / (mutual_info_scores.max() - mutual_info_scores.min())

# **已有的 ESD 剪枝比例**
esd_pruning_ratios = torch.tensor([
    0.5704, 0.6176, 0.6315, 0.6307, 0.6528, 0.6482, 0.6300, 0.5921,
    0.5973, 0.5680, 0.5870, 0.5893, 0.5989, 0.6108, 0.6187, 0.6681,
    0.6586, 0.7156, 0.7905, 0.7437, 0.7946, 0.8248, 0.7700, 0.7629,
    0.8121, 0.8520, 0.8312, 0.8414, 0.7869, 0.8296, 0.8414, 0.7317
], device="cuda")

# **🔥 融合 ESD 重要性 & 层间关系**
alpha = 0.5  # Wasserstein 影响权重
beta = 0.5   # 互信息影响权重

adjusted_pruning_ratios = esd_pruning_ratios.clone()
for i in range(num_layers - 1):
    adjusted_pruning_ratios[i] *= (1 + alpha * wasserstein_scores[i] - beta * mutual_info_scores[i])

# **归一化**
adjusted_pruning_ratios = adjusted_pruning_ratios / torch.sum(adjusted_pruning_ratios) * torch.sum(esd_pruning_ratios)

# **输出**
print("✅ 原始剪枝比例:", esd_pruning_ratios.cpu().numpy())
print("✅ 调整后剪枝比例:", adjusted_pruning_ratios.cpu().numpy())
