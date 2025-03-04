import torch
from transformers import AutoModelForCausalLM, LlamaTokenizer

# ✅ 自动适配多 GPU
device_count = torch.cuda.device_count()
device_map = {i: f"cuda:{i}" for i in range(device_count)}
print(f"🚀 Using {device_count} GPUs: {device_map}")

# ✅ 加载 LLaMA-7B
cache_dir = "/root/autodl-tmp/llm_weights"
model = AutoModelForCausalLM.from_pretrained(
    "pinkmanlove/llama-7b-hf",
    cache_dir=cache_dir,
    torch_dtype=torch.float16,
    device_map="auto",
)

tokenizer = LlamaTokenizer.from_pretrained("HuggingFaceM4/llama-7b-tokenizer")

# ✅ 存储梯度 × 激活值
grad_activation_scores = {}

def forward_hook(layer_idx):
    """存储前向传播的激活值"""
    def hook(module, input, output):
        layer_name = f"LlamaDecoderLayer_{layer_idx}"  # ✅ 直接存层数索引
        hidden_states = output[0] if isinstance(output, tuple) else output  # ✅ 兼容 tuple 输出
        grad_activation_scores[layer_name] = {"activation": hidden_states.detach()}
    return hook

def backward_hook(layer_idx):
    """计算梯度 × 激活值"""
    def hook(module, grad_input, grad_output):
        layer_name = f"LlamaDecoderLayer_{layer_idx}"

        gradient = grad_output[0].detach() if isinstance(grad_output, tuple) else grad_output.detach()
        activation = grad_activation_scores[layer_name]["activation"]

        # ✅ 确保梯度和激活值在同一设备
        if gradient.device != activation.device:
            activation = activation.to(gradient.device)

        # 🚀 计算贡献度
        contribution = (gradient * activation).mean().item()
        grad_activation_scores[layer_name]["contribution"] = contribution

        print(f"✅ Processed {layer_name}: Contribution={contribution:.6f}")
    return hook

# ✅ 绑定 Hooks (修正作用范围)
hooks = []
for idx, layer in enumerate(model.model.layers):
    hooks.append(layer.register_forward_hook(forward_hook(idx)))
    hooks.append(layer.register_full_backward_hook(backward_hook(idx)))  # ✅ 兼容 `accelerate`

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

# ✅ 提取并排序贡献度
sorted_grad_activations = sorted(
    [(name, data["contribution"]) for name, data in grad_activation_scores.items() if "contribution" in data],
    key=lambda x: -x[1]
)

# ✅ 打印结果
print("\n🚀 **梯度 × 激活值 贡献度（按重要性排序）** 🚀\n")
for layer, score in sorted_grad_activations:
    print(f"{layer}: Contribution={score:.6f}")
