import torch
import torch.nn.functional as F
import numpy as np
from transformers import LlamaModel, AutoTokenizer

# **🔥 确保模型用多个 GPU**
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

# **🔥 逐块计算 Wasserstein 距离，防止显存溢出**
def wasserstein_distance_torch(H1, H2, eps=1e-3, max_iter=50, chunk_size=512):
    """
    计算 Sinkhorn-Wasserstein 距离，逐块计算 Cost 矩阵，减少显存占用
    """
    H1 = H1.to(torch.float16)  # **降低精度减少显存占用**
    H2 = H2.to(torch.float16)  

    n, d = H1.shape
    m, _ = H2.shape

    # **初始化 cost_matrix**
    cost_matrix = torch.zeros(n, m, dtype=torch.float16, device=H1.device)

    # 🔥 **分块计算 Cost 矩阵**
    for i in range(0, n, chunk_size):
        for j in range(0, m, chunk_size):
            sub_H1 = H1[i : i + chunk_size]
            sub_H2 = H2[j : j + chunk_size]
            cost_matrix[i : i + chunk_size, j : j + chunk_size] = torch.cdist(sub_H1, sub_H2, p=2).pow(2)

    torch.cuda.empty_cache()  # **释放显存**
    
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

# **🔥 计算 Wasserstein**
layerwise_wasserstein = []

for layer_idx in range(num_layers - 1):
    H = hidden_states[layer_idx][0].to(torch.float16)
    H_next = hidden_states[layer_idx + 1][0].to(torch.float16)

    # **🔥 使用 Chunk 计算 Wasserstein，避免 OOM**
    wd = wasserstein_distance_torch(H, H_next, chunk_size=256)  
    layerwise_wasserstein.append(wd.item())

# **🔥 归一化 Wasserstein**
wasserstein_scores = torch.tensor(layerwise_wasserstein, device="cuda")
wasserstein_scores = (wasserstein_scores - wasserstein_scores.min()) / (wasserstein_scores.max() - wasserstein_scores.min())

# **🔥 输出**
print("✅ Wasserstein 计算完成")
print("Wasserstein Scores:", wasserstein_scores.cpu().numpy())
