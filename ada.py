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

# -----------------------------
# 2. 构造一个 dummy input
# -----------------------------
# 生成一个 [batch_size, seq_length] 的随机 token id 张量，范围 [0, vocab_size)
# 注意：模型配置中应有 vocab_size 属性
input_ids = torch.randint(0, model.config.vocab_size, (1, 16)).to(model.device)

# -----------------------------
# 3. 前向传播并计算损失
# -----------------------------
# 对于因果语言模型，我们可以使用 labels=input_ids 计算交叉熵损失
output = model(input_ids=input_ids, labels=input_ids)
loss = output.loss

# -----------------------------
# 4. 定义辅助函数：获取层参数、计算 Hessian Trace（使用 Hutchinson 方法）和梯度范数
# -----------------------------
def get_layer_parameters(layer):
    return [p for p in layer.parameters() if p.requires_grad]

def compute_hessian_trace(layer, loss, n_samples=1, eps=1e-2):
    """
    利用 Hutchinson 方法计算该层参数的 Hessian Trace 近似值：
        Tr(H) ≈ E[ v^T H v ]
    其中 v 为随机向量（这里采用高斯分布）。
    """
    params = get_layer_parameters(layer)
    if not params:
        return 0.0
    trace_estimates = []
    for _ in range(n_samples):
        vs = [torch.randn_like(p) for p in params]
        # 计算梯度，保持计算图
        grad_params = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
        # 计算点积：sum_i <grad_i, v_i>
        dot = sum(torch.sum(g * v) for g, v in zip(grad_params, vs))
        # 计算 Hessian-Vector 产品：H v = grad(dot)
        hvps = torch.autograd.grad(dot, params, retain_graph=True)
        # 计算 v^T H v：累加所有参数的 v_i * (H v)_i
        hvp_dot = sum(torch.sum(v * hvp) for v, hvp in zip(vs, hvps))
        trace_estimates.append(hvp_dot.item())
    return sum(trace_estimates) / len(trace_estimates)

def compute_gradient_norm(layer, loss):
    """
    计算该层所有参数的梯度范数（L2），并取和。
    """
    params = get_layer_parameters(layer)
    if not params:
        return 0.0
    grad_params = torch.autograd.grad(loss, params, retain_graph=True)
    norm = sum(torch.norm(g, p=2).item() for g in grad_params)
    return norm

# -----------------------------
# 5. 针对每个 Transformer 层计算 Hessian Trace 和梯度范数
# -----------------------------
hessian_scores = {}
gradnorm_scores = {}

# 假设 Transformer 层存放在 model.model.layers
for i, layer in enumerate(model.model.layers):
    ht = compute_hessian_trace(layer, loss, n_samples=1, eps=1e-2)
    gn = compute_gradient_norm(layer, loss)
    hessian_scores[str(i)] = ht
    gradnorm_scores[str(i)] = gn

print("Hessian scores per layer:", hessian_scores)
print("Gradient norm scores per layer:", gradnorm_scores)

# -----------------------------
# 6. 对两个指标归一化
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
# 7. 融合一阶和二阶信息生成最终重要性分数
# -----------------------------
# alpha 控制 Hessian 信息的权重，建议取 0.7（可根据验证结果调整）
alpha = 0.7
importance_scores = {}
for k in norm_hessian.keys():
    # 重要性分数 = alpha * (归一化 Hessian) + (1 - alpha) * (归一化梯度)
    importance_scores[k] = alpha * norm_hessian[k] + (1 - alpha) * norm_gradnorm[k]

print("Final importance scores per layer:")
for k in sorted(importance_scores, key=lambda x: int(x)):
    print(f"Layer {k}: {importance_scores[k]:.4f}")

# -----------------------------
# 8. (可选) 根据层重要性生成剪枝比例
# -----------------------------
# 假设全局剪枝率为 p_base（例如60%），重要性越高的层剪枝比例应越低
p_base = 0.6
pruning_ratios = {}
for k, imp in importance_scores.items():
    # 可以简单设置剪枝比例为： p_i = p_base * (1 - importance)
    pruning_ratios[k] = p_base * (1 - importance_scores[k])
    
print("Suggested per-layer pruning ratios:")
for k in sorted(pruning_ratios, key=lambda x: int(x)):
    print(f"Layer {k}: {pruning_ratios[k]:.4f}")

# 以上重要性分数和剪枝比例可以作为 SparseGPT、Wanda 等剪枝方法的输入指导剪枝决策。
