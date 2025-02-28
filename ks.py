import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, LlamaTokenizer

# 设置设备
device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载 LLaMA 模型和 tokenizer
cache_dir = "/root/autodl-tmp/llm_weights"
model = AutoModelForCausalLM.from_pretrained(
    "pinkmanlove/llama-7b-hf",
    cache_dir=cache_dir,
    device_map="auto",
    torch_dtype=torch.float16
)
model.to(device)

tokenizer = LlamaTokenizer.from_pretrained("HuggingFaceM4/llama-7b-tokenizer")

# 准备输入
text = "Hello, this is a test input for importance computation."
inputs = tokenizer(text, return_tensors="pt").to(device)

# 启用梯度计算
for param in model.parameters():
    param.requires_grad = True

# 计算前向传播
outputs = model(**inputs)
loss = outputs.logits.mean()  # 这里用 logits 的均值作为损失函数，仅用于梯度计算
loss.backward()

# 计算梯度范数
module_keys = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
num_layers = len(model.model.layers)
grad_norms = {key: np.zeros(num_layers) for key in module_keys}

for i in range(num_layers):
    layer_key = f"model.layers.{i}"
    for key in module_keys:
        # 找到该层中属于 key 的所有参数，并计算梯度范数
        layer_params = [param.grad for name, param in model.named_parameters()
                        if layer_key in name and key in name and param.grad is not None]
        if len(layer_params) > 0:
            grad_norms[key][i] = np.mean([torch.norm(p, p=2).item() for p in layer_params])
        else:
            grad_norms[key][i] = 0  # 如果该层没有这个模块，则设为 0

# 归一化梯度范数（防止极端值影响）
for key in grad_norms:
    max_val = np.max(grad_norms[key])
    if max_val > 0:
        grad_norms[key] /= max_val

print(grad_norms)
# 保存结果
np.save("llama_component_grad_norms.npy", grad_norms)
print("Gradient norm importance scores saved to llama_component_grad_norms.npy")
