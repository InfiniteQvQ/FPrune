import torch
from transformers import AutoModelForCausalLM

# -----------------------------
# 1. 加载 LLaMA 7B 模型（从本地路径）
# -----------------------------
cache_dir = "/root/autodl-tmp/llm_weights"
model = AutoModelForCausalLM.from_pretrained(
    "pinkmanlove/llama-7b-hf",
    cache_dir=cache_dir,
    device_map="auto",
    torch_dtype=torch.float16
)
model.eval()
# 为了数值稳定，转为 FP32 进行 Hessian 计算
model = model.float()

# -----------------------------
# 2. 构造一个 dummy input 并计算 loss
# -----------------------------
input_ids = torch.randint(0, model.config.vocab_size, (1, 16)).to(model.device)
output = model(input_ids=input_ids, labels=input_ids)
loss = output.loss

# -----------------------------
# 3. 辅助函数：获取层中需要计算的参数（仅考虑权重，不含偏置）
# -----------------------------
def get_layer_parameters(layer):
    # 只选取至少是二维的权重参数（例如矩阵），跳过一维的bias等
    return [p for p in layer.parameters() if p.requires_grad and p.dim() >= 2]

# -----------------------------
# 4. 利用 Hutchinson 方法近似计算单个参数的 Hessian 对角线
# -----------------------------
def compute_hessian_diag(p, loss, n_samples=3, eps=1e-2):
    """
    对参数 p，利用 Hutchinson 方法估计 Hessian 对角线的元素。
    我们采用 Rademacher 随机向量（取值 ±1），多次采样取平均，并返回一个与 p 同形状的估计张量。
    """
    diag_estimates = []
    for _ in range(n_samples):
        # 生成 Rademacher 随机向量：每个元素取 ±1
        v = torch.randint(0, 2, p.shape, device=p.device).float() * 2 - 1
        # 计算 p 关于 loss 的梯度（需保持计算图）
        grad_p = torch.autograd.grad(loss, p, create_graph=True, retain_graph=True)[0]
        # 计算 Hessian-Vector 产品：H*v
        hvp = torch.autograd.grad(grad_p, p, grad_outputs=v, retain_graph=True)[0]
        # Hutchinson 近似： v * (H*v) 近似等于 p 对应的 Hessian 对角元素
        diag_estimates.append(v * hvp)
    # 取平均
    diag_mean = torch.mean(torch.stack(diag_estimates), dim=0)
    return diag_mean

# -----------------------------
# 5. 对每个 Transformer 层计算重要性分数
# -----------------------------
def compute_layer_importance(layer, loss, n_samples=3, eps=1e-2):
    """
    对给定层计算重要性分数：
      I_layer = sum_{p in layer} sum( p^2 * |diag_H| )
    其中 diag_H 是通过 Hutchinson 方法估计的 Hessian 对角线。
    """
    params = get_layer_parameters(layer)
    layer_importance = 0.0
    for p in params:
        # 计算 p^2（逐元素平方）
        weight_square = p.detach() ** 2
        # 计算 Hessian 对角线近似（取多次采样平均），并取绝对值
        hessian_diag = compute_hessian_diag(p, loss, n_samples=n_samples, eps=eps)
        layer_importance += torch.sum(weight_square * hessian_diag.abs()).item()
    return layer_importance

# -----------------------------
# 6. 遍历所有 Transformer 层（假设在 model.model.layers 中）计算重要性
# -----------------------------
layer_importance_scores = {}
for i, layer in enumerate(model.model.layers):
    imp = compute_layer_importance(layer, loss, n_samples=3, eps=1e-2)
    layer_importance_scores[str(i)] = imp

print("Raw layer importance scores (unnormalized):")
for k in sorted(layer_importance_scores, key=lambda x: int(x)):
    print(f"Layer {k}: {layer_importance_scores[k]:.4f}")

# -----------------------------
# 7. 对所有层的重要性归一化
# -----------------------------
def normalize_dict(d, eps=1e-8):
    values = list(d.values())
    min_val = min(values)
    max_val = max(values)
    norm = {k: (v - min_val) / (max_val - min_val + eps) for k, v in d.items()}
    return norm

norm_importance = normalize_dict(layer_importance_scores)
print("Normalized layer importance scores:")
norm_list = [norm_importance[k] for k in sorted(norm_importance, key=lambda x: int(x))]
print(norm_list)



