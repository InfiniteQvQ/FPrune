import torch
import numpy as np
from transformers import AutoModelForCausalLM, LlamaTokenizer
import torch.nn as nn

# 1. 设定设备：使用所有可用 GPU（DataParallel）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_gpus = torch.cuda.device_count()  # 获取 GPU 数量
print(f"Using {num_gpus} GPUs.")

# 2. 加载 LLaMA 模型（不使用 device_map="auto"，手动移动到 GPU）
cache_dir = "/root/autodl-tmp/llm_weights"  # 修改为你的模型路径
model = AutoModelForCausalLM.from_pretrained(
    "pinkmanlove/llama-7b-hf",
    cache_dir=cache_dir,
    torch_dtype=torch.float16
)

# 3. 使用 DataParallel 在多个 GPU 之间分配模型
if num_gpus > 1:
    model = nn.DataParallel(model)  # 多 GPU 计算
model.to(device)

# 4. 加载 Tokenizer
tokenizer = LlamaTokenizer.from_pretrained("HuggingFaceM4/llama-7b-tokenizer")

# 5. 构造输入，并确保输入在 GPU 上
text = "Hello, this is a test input for importance computation."
inputs = tokenizer(text, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

# 6. 启用梯度计算
for param in model.parameters():
    param.requires_grad = True

# 7. 前向传播及反向传播（计算梯度）
outputs = model(**inputs)
loss = outputs.logits.mean()  # 用 logits 的均值作为损失
loss.backward()

# 8. 计算每一层中各模块的梯度范数
module_keys = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
model_layers = model.module.model.layers if num_gpus > 1 else model.model.layers  # 兼容单 GPU & 多 GPU
num_layers = len(model_layers)
grad_norms = {key: np.zeros(num_layers) for key in module_keys}

for i in range(num_layers):
    layer_key = f"model.layers.{i}"
    for key in module_keys:
        # 在 model.module.named_parameters() 中查找当前层中包含该模块名称的参数
        layer_params = [
            param.grad.detach().cpu() for name, param in model.named_parameters()
            if layer_key in name and key in name and param.grad is not None
        ]
        if layer_params:
            grad_norms[key][i] = np.mean([torch.norm(p, p=2).item() for p in layer_params])
        else:
            grad_norms[key][i] = 0

# 9. 对每个模块的梯度范数进行归一化，防止显存溢出
for key in grad_norms:
    max_val = np.max(grad_norms[key])
    if max_val > 0:
        grad_norms[key] /= max_val

# 10. 打印并保存结果（避免显存占用过多）
print("Normalized Gradient Norms for each module:")
print(grad_norms)

np.save("llama_component_grad_norms.npy", grad_norms)
print("Gradient norm importance scores saved to llama_component_grad_norms.npy")
