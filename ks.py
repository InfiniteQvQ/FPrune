import torch
import numpy as np
from transformers import AutoModelForCausalLM, LlamaTokenizer
import torch.nn as nn

# 1. 设定设备：使用 DistributedDataParallel（DDP）代替 DataParallel
torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_gpus = torch.cuda.device_count()  # 获取 GPU 数量
print(f"Using {num_gpus} GPUs.")

# 2. 加载 LLaMA 模型（启用 gradient_checkpointing 以减少显存占用）
cache_dir = "/root/autodl-tmp/llm_weights"  # 修改为你的模型路径
model = AutoModelForCausalLM.from_pretrained(
    "pinkmanlove/llama-7b-hf",
    cache_dir=cache_dir,
    torch_dtype=torch.float16
)

# **启用 Gradient Checkpointing，减少显存占用**
model.gradient_checkpointing_enable()

# **使用 DDP（多 GPU 训练）**
if num_gpus > 1:
    model = nn.parallel.DistributedDataParallel(model, device_ids=[i for i in range(num_gpus)])
model.to(device)

# 3. 加载 Tokenizer
tokenizer = LlamaTokenizer.from_pretrained("HuggingFaceM4/llama-7b-tokenizer")

# 4. 构造输入，并确保输入在 GPU 上
text = "Hello, this is a test input for importance computation."
inputs = tokenizer(text, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

# 5. 仅计算部分层的梯度，避免显存溢出
for name, param in model.named_parameters():
    if "layers.0" in name or "layers.1" in name:  # 只计算前 2 层
        param.requires_grad = True
    else:
        param.requires_grad = False  # 其他层不计算梯度，减少显存占用

# 6. 前向传播及反向传播（计算梯度）
outputs = model(**inputs)
loss = outputs.logits.mean()  # 用 logits 的均值作为损失
loss.backward()

# 7. 计算每一层中各模块的梯度范数
module_keys = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
model_layers = model.module.model.layers if num_gpus > 1 else model.model.layers  # 兼容单 GPU & 多 GPU
num_layers = len(model_layers)
grad_norms = {key: np.zeros(num_layers) for key in module_keys}

for i in range(num_layers):
    layer_key = f"model.layers.{i}"
    for key in module_keys:
        # 在 model.named_parameters() 中查找当前层的参数
        layer_params = [
            param.grad.detach().cpu() for name, param in model.named_parameters()
            if layer_key in name and key in name and param.grad is not None
        ]
        if layer_params:
            grad_norms[key][i] = np.mean([torch.norm(p, p=2).item() for p in layer_params])
        else:
            grad_norms[key][i] = 0

# 8. 归一化梯度范数，防止显存溢出
for key in grad_norms:
    max_val = np.max(grad_norms[key])
    if max_val > 0:
        grad_norms[key] /= max_val

# 9. 清理 CUDA 缓存，减少显存占用
torch.cuda.empty_cache()

# 10. 打印并保存结果（避免显存占用过多）
print("Normalized Gradient Norms for each module:")
print(grad_norms)

np.save("llama_component_grad_norms.npy", grad_norms)
print("Gradient norm importance scores saved to llama_component_grad_norms.npy")
