import torch
import numpy as np
from transformers import AutoModelForCausalLM, LlamaTokenizer

# 1. 检测 GPU
torch.cuda.empty_cache()
num_gpus = torch.cuda.device_count()
assert num_gpus >= 2, "You need at least 2 GPUs for this setup!"
print(f"Using {num_gpus} GPUs.")

# 2. 加载 LLaMA-7B 模型（Float16 省显存）
cache_dir = "/root/autodl-tmp/llm_weights"  # 你的模型存放路径
model = AutoModelForCausalLM.from_pretrained(
    "pinkmanlove/llama-7b-hf",
    cache_dir=cache_dir,
    torch_dtype=torch.float16
)

# 启用 Gradient Checkpointing，减少显存占用
model.gradient_checkpointing_enable()

# 3. **手动分配 Transformer 层到不同 GPU**
device_0 = "cuda:0"  # 第一张 GPU
device_1 = "cuda:1"  # 第二张 GPU

# 把 LLaMA 32 层的前 16 层放到 GPU 0，后 16 层放到 GPU 1
for i, layer in enumerate(model.model.layers):
    if i < 16:
        layer.to(device_0)
    else:
        layer.to(device_1)

# 4. 加载 Tokenizer
tokenizer = LlamaTokenizer.from_pretrained("HuggingFaceM4/llama-7b-tokenizer")

# 5. **输入数据**
text = "Hello, this is a test input for importance computation."
inputs = tokenizer(text, return_tensors="pt")
inputs = {k: v.to(device_0) for k, v in inputs.items()}  # **输入放在 GPU 0**

# 6. **前向传播（自动跨 GPU 计算）**
with torch.cuda.amp.autocast():  # 启用混合精度，减少显存
    outputs = model(**inputs)

# 7. 计算损失
loss = outputs.logits.mean()
loss.backward()

# 8. **梯度计算（提取每层的梯度范数）**
module_keys = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
model_layers = model.model.layers
num_layers = len(model_layers)
grad_norms = {key: np.zeros(num_layers) for key in module_keys}

for i in range(num_layers):
    layer_key = f"model.layers.{i}"
    for key in module_keys:
        # 获取当前层的梯度
        layer_params = [
            param.grad.detach().cpu() for name, param in model.named_parameters()
            if layer_key in name and key in name and param.grad is not None
        ]
        if layer_params:
            grad_norms[key][i] = np.mean([torch.norm(p, p=2).item() for p in layer_params])
        else:
            grad_norms[key][i] = 0

# 9. **归一化梯度范数**
for key in grad_norms:
    max_val = np.max(grad_norms[key])
    if max_val > 0:
        grad_norms[key] /= max_val

# 10. 释放显存，减少占用
torch.cuda.empty_cache()

# 11. **保存结果**
print("Normalized Gradient Norms for each module:")
print(grad_norms)

np.save("llama_component_grad_norms.npy", grad_norms)
print("Gradient norm importance scores saved to llama_component_grad_norms.npy")
