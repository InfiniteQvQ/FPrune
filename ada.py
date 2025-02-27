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

# 为了数值稳定性，转换模型为 FP32（仅用于 Hessian 计算）
model = model.float()

# -----------------------------
# 2. 构造 dummy input 并计算损失
# -----------------------------
input_ids = torch.randint(0, model.config.vocab_size, (1, 16)).to(model.device)
output = model(input_ids=input_ids, labels=input_ids)
loss = output.loss

# -----------------------------
# 3. 辅助函数：获取层参数、计算 Hessian Trace（Hutchinson方法）和梯度范数
# -----------------------------
def get_layer_parameters(layer):
    # 只计算权重参数（要求参数至少二维，避免偏置等一维参数）
    return [p for p in layer.parameters() if p.requires_grad and p.dim() >= 2]

def compute_hessian_trace(layer, loss, n_samples=3, eps=1e-1):
    """
    利用 Hutchinson 方法计算该层参数的 Hessian Trace 近似值：
      Tr(H) ≈ E[ v^T H v ]
    其中 v 为随机向量，使用 n_samples 次采样取平均，并返回绝对值以避免负值。
    """
    params = get_layer_parameters(layer)
    if not params:
        return 0.0
    trace_estimates = []
    for _ in range(n_samples):
        vs = [torch.randn_like(p) for p in params]
        grad_params = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
        dot = sum(torch.sum(g * v) for g, v in zip(grad_params, vs))
        hvps = torch.autograd.grad(dot, params, retain_graph=True)
        hvp_dot = sum(torch.sum(v * hvp) for v, hvp in zip(vs, hvps))
        trace_estimates.append(hvp_dot.item())
    trace_value = sum(trace_estimates) / len(trace_estimates)
    return abs(trace_value)  # 取绝对值

def compute_gradient_norm(layer, loss):
    """
    计算该层所有参数的梯度L2范数和（作为一阶信息）。
    """
    params = get_layer_parameters(layer)
    if not params:
        return 0.0
    grad_params = torch.autograd.grad(loss, params, retain_graph=True)
    norm_val = sum(torch.norm(g, p=2).item() for g in grad_params)
    return norm_val

# -----------------------------
# 4. 针对每个 Transformer 层计算 Hessian Trace 和梯度范数
# -----------------------------
hessian_scores = {}
gradnorm_scores = {}

# 假设 Transformer 层存放在 model.model.layers 中
for i, layer in enumerate(model.model.layers):
    ht = compute_hessian_trace(layer, loss, n_samples=3, eps=1e-1)
    gn = compute_gradient_norm(layer, loss)
    hessian_scores[str(i)] = ht
    gradnorm_scores[str(i)] = gn

print("Hessian scores per layer:", hessian_scores)
print("Gradient norm scores per layer:", gradnorm_scores)

# -----------------------------
# 5. 对两个指标归一化
# -----------------------------
def normalize_dict(d, eps=1e-8):
    values = list(d.values())
    min_val = min(values)
    max_val = max(values)
    norm = {k: (v - min_val) / (max_val - min_val + eps) for k, v in d.items()}
    return norm

norm_hessian = normalize_dict(hessian_scores)
norm_gradnorm = normalize_dict(gradnorm_scores)

print("Normalized Hessian scores:", norm_hessian)
print("Normalized Gradient Norm scores:", norm_gradnorm)

# -----------------------------
# 6. 融合一阶和二阶信息生成最终重要性分数
# -----------------------------
# alpha 控制二阶信息的权重，这里建议取 0.7（可根据验证结果调整）
alpha = 0.7
importance_scores = {}
for k in norm_hessian.keys():
    importance_scores[k] = alpha * norm_hessian[k] + (1 - alpha) * norm_gradnorm[k]

print("Final importance scores per layer:")
for k in sorted(importance_scores, key=lambda x: int(x)):
    print(f"Layer {k}: {importance_scores[k]:.4f}")

# -----------------------------
# 7. 根据层重要性生成建议剪枝比例
# -----------------------------
# 假设全局剪枝率为 p_base（例如 60%），重要性越高的层剪枝比例应越低
p_base = 0.7
pruning_ratios = {}
for k, imp in importance_scores.items():
    pruning_ratios[k] = p_base * (1 - imp)
    
print("Suggested per-layer pruning ratios:")
for k in sorted(pruning_ratios, key=lambda x: int(x)):
    print(f"Layer {k}: {pruning_ratios[k]:.4f}")
