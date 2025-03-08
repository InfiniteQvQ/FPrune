import torch
import torch.nn.functional as F
import numpy as np
from transformers import LlamaModel, AutoTokenizer

# **加载 Llama 7B**
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

# **计算 Wasserstein 距离 (Sinkhorn-Knopp 迭代)**
def wasserstein_distance_torch(H1, H2, eps=1e-3, max_iter=50):
    H1 = H1.view(-1, 1).float()
    H2 = H2.view(-1, 1).float()
    
    # 计算 Cost 矩阵 (欧几里得距离)
    C = torch.cdist(H1, H2, p=2).pow(2)  # (N, M)
    
    # 初始化分布
    a = torch.ones(H1.size(0), device=H1.device) / H1.size(0)
    b = torch.ones(H2.size(0), device=H2.device) / H2.size(0)
    
    # Sinkhorn-Knopp 迭代
    u = torch.zeros_like(a)
    v = torch.zeros_like(b)
    
    for _ in range(max_iter):
        u = -torch.logsumexp((-C + v[None, :]) / eps, dim=1) + torch.log(a)
        v = -torch.logsumexp((-C + u[:, None]) / eps, dim=0) + torch.log(b)
    
    return (u[:, None] + v[None, :] - C).exp().sum()

# **计算 GPU 互信息**
def mutual_information_torch(H1, H2, num_bins=256):
    H1 = H1.view(-1).float()
    H2 = H2.view(-1).float()

    # 计算直方图
    hist2d = torch.histogram2d(H1, H2, bins=num_bins, weight=None, range=None)
    
    # 归一化为联合概率
    P_xy = hist2d.hist / torch.sum(hist2d.hist)
    
    # 计算边际概率
    P_x = torch.sum(P_xy, dim=1, keepdim=True)
    P_y = torch.sum(P_xy, dim=0, keepdim=True)
    
    # 计算互信息
    P_xy_nonzero = P_xy[P_xy > 0]
    P_x_nonzero = P_x[P_x > 0]
    P_y_nonzero = P_y[P_y > 0]
    
    I_xy = (P_xy_nonzero * torch.log(P_xy_nonzero / (P_x_nonzero * P_y_nonzero))).sum()
    return I_xy

# **计算 Wasserstein 和 互信息**
layerwise_wasserstein = []
layerwise_mutual_info = []

for layer_idx in range(num_layers - 1):
    H = hidden_states[layer_idx][0].to(torch.float32)  # 取 batch=0, 转为 float32 避免 underflow
    H_next = hidden_states[layer_idx + 1][0].to(torch.float32)

    # **使用 GPU 计算 Wasserstein 距离**
    wd = wasserstein_distance_torch(H, H_next)
    layerwise_wasserstein.append(wd.item())

    # **使用 GPU 计算互信息**
    mi = mutual_information_torch(H, H_next)
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

# **融合 ESD 重要性 和 层间信息**
alpha = 0.5  # Wasserstein 影响权重
beta = 0.5   # 互信息影响权重

adjusted_pruning_ratios = esd_pruning_ratios.clone()
for i in range(num_layers - 1):
    adjusted_pruning_ratios[i] *= (1 + alpha * wasserstein_scores[i] - beta * mutual_info_scores[i])

# **归一化，使剪枝比例不变**
adjusted_pruning_ratios = adjusted_pruning_ratios / torch.sum(adjusted_pruning_ratios) * torch.sum(esd_pruning_ratios)

# **输出调整后的剪枝比例**
print("原始剪枝比例:", esd_pruning_ratios.cpu().numpy())
print("调整后剪枝比例:", adjusted_pruning_ratios.cpu().numpy())
