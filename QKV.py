import torch
from transformers import AutoModelForCausalLM, LlamaTokenizer

# 🔹 加载 LLaMA-7B 模型
cache_dir = "/root/autodl-tmp/llm_weights"
model = AutoModelForCausalLM.from_pretrained(
    "pinkmanlove/llama-7b-hf",
    cache_dir=cache_dir,
    device_map="auto",
    torch_dtype=torch.float16
)
tokenizer = LlamaTokenizer.from_pretrained("HuggingFaceM4/llama-7b-tokenizer")

# 🔹 存储梯度 × 激活值
grad_activation_scores = {}

def forward_hook(module, input, output):
    """ 存储前向传播的激活值 """
    layer_name = module._get_name()
    
    # 🔹 兼容 tuple 输出（通常 LlamaDecoderLayer 返回多个值）
    if isinstance(output, tuple):
        hidden_states = output[0]  # 取 hidden_states
    else:
        hidden_states = output

    grad_activation_scores[layer_name] = {"activation": hidden_states.detach()}

def backward_hook(module, grad_input, grad_output):
    """ 计算梯度 × 激活值 """
    layer_name = module._get_name()
    
    # 🔹 兼容 tuple 的 grad_output
    if isinstance(grad_output, tuple):
        gradient = grad_output[0].detach()
    else:
        gradient = grad_output.detach()

    activation = grad_activation_scores[layer_name]["activation"]
    
    # 计算贡献度
    contribution = (gradient * activation).mean().item()
    grad_activation_scores[layer_name]["contribution"] = contribution

# 🔹 绑定前向 & 反向传播 Hook
hooks = []
for layer_id, layer in enumerate(model.model.layers):
    fwd_hook = layer.register_forward_hook(forward_hook)
    bwd_hook = layer.register_full_backward_hook(backward_hook)
    hooks.extend([fwd_hook, bwd_hook])

# 🔹 运行模型并计算梯度
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
text = "Artificial Intelligence is transforming the world with LLaMA-7B."
inputs = tokenizer(text, return_tensors="pt").to(device)

# 🔹 计算 Loss 并反向传播
outputs = model(**inputs, labels=inputs["input_ids"])
loss = outputs.loss
loss.backward()

# 🔹 释放 Hooks
for hook in hooks:
    hook.remove()

# 🔹 提取并排序贡献度
sorted_grad_activations = sorted(
    [(name, data["contribution"]) for name, data in grad_activation_scores.items() if "contribution" in data],
    key=lambda x: -x[1]
)

# 🔹 打印结果
print("\n🚀 **梯度 × 激活值 贡献度（按重要性排序）** 🚀\n")
for layer, score in sorted_grad_activations:
    print(f"{layer}: Contribution={score:.6f}")
