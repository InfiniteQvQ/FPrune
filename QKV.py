import torch
from transformers import AutoModelForCausalLM, LlamaTokenizer

# ✅ 自动多 GPU 映射
device_count = torch.cuda.device_count()
device_map = {i: f"cuda:{i}" for i in range(device_count)}  
print(f"🚀 Using {device_count} GPUs: {device_map}")

# ✅ 加载 LLaMA-7B
cache_dir = "/root/autodl-tmp/llm_weights"
model = AutoModelForCausalLM.from_pretrained(
    "pinkmanlove/llama-7b-hf",
    cache_dir=cache_dir,
    torch_dtype=torch.float16,
    device_map="auto",  # 🚀 让 Hugging Face 自动分配多个 GPU
)

tokenizer = LlamaTokenizer.from_pretrained("HuggingFaceM4/llama-7b-tokenizer")

# ✅ 存储梯度 × 激活值
grad_activation_scores = {}

def forward_hook(module, input, output):
    """存储前向传播的激活值"""
    layer_name = module._get_name() + f"_{id(module)}"  # ✅ 确保每个 LlamaDecoderLayer 都有唯一名字
    
    # ✅ 兼容 tuple 输出
    if isinstance(output, tuple):
        hidden_states = output[0]
    else:
        hidden_states = output

    # ✅ 确保存储在当前计算 GPU
    grad_activation_scores[layer_name] = {"activation": hidden_states.detach().to(hidden_states.device)}

def backward_hook(module, grad_input, grad_output):
    """计算梯度 × 激活值"""
    layer_name = module._get_name() + f"_{id(module)}"

    # ✅ 兼容 tuple 输出
    if isinstance(grad_output, tuple):
        gradient = grad_output[0].detach()
    else:
        gradient = grad_output.detach()

    activation = grad_activation_scores[layer_name]["activation"]

    # ✅ 确保梯度和激活值在同一个 GPU
    if gradient.device != activation.device:
        activation = activation.to(gradient.device)

    # 🚀 计算贡献度
    contribution = (gradient * activation).mean().item()

    # ✅ 确保存到 `cuda:0`
    grad_activation_scores[layer_name]["contribution"] = torch.tensor(contribution, device="cuda:0")

    print(f"✅ Processed {layer_name}: Contribution={contribution:.6f}")

# ✅ 绑定 Hooks
hooks = []
for idx, layer in enumerate(model.model.layers):
    layer_name = f"LlamaDecoderLayer_{idx}"
    fwd_hook = layer.register_forward_hook(forward_hook)
    bwd_hook = layer.register_full_backward_hook(backward_hook)  # ✅ `register_full_backward_hook`
    hooks.extend([fwd_hook, bwd_hook])

# ✅ 运行模型
text = "Artificial Intelligence is transforming the world with LLaMA-7B."
inputs = tokenizer(text, return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}  # ✅ 发送到 `model` 设备

# ✅ 计算 Loss 并反向传播
outputs = model(**inputs, labels=inputs["input_ids"])
loss = outputs.loss
loss.backward()

# ✅ 释放 Hooks
for hook in hooks:
    hook.remove()

# ✅ 统一收集 `梯度 × 激活值` 数据到 `cuda:0`
for layer_name, data in grad_activation_scores.items():
    if "contribution" in data:
        grad_activation_scores[layer_name]["contribution"] = data["contribution"].to("cuda:0")

# ✅ 提取并排序贡献度
sorted_grad_activations = sorted(
    [(name, data["contribution"].item()) for name, data in grad_activation_scores.items() if "contribution" in data],
    key=lambda x: -x[1]
)

# ✅ 打印结果
print("\n🚀 **梯度 × 激活值 贡献度（按重要性排序）** 🚀\n")
for layer, score in sorted_grad_activations:
    print(f"{layer}: Contribution={score:.6f}")
