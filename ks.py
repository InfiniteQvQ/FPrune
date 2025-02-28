import torch
import numpy as np
from transformers import AutoModelForCausalLM, LlamaTokenizer

# **1. 设定设备**
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# **2. 加载 LLaMA 模型**
cache_dir = "/root/autodl-tmp/llm_weights"  # 修改为你的模型路径
model = AutoModelForCausalLM.from_pretrained(
    "pinkmanlove/llama-7b-hf",
    cache_dir=cache_dir,
    torch_dtype=torch.float16
)
model.to(device)  # **确保模型在 GPU**

# **3. 加载 Tokenizer**
tokenizer = LlamaTokenizer.from_pretrained("HuggingFaceM4/llama-7b-tokenizer")

# **4. 构造输入**
text = "Hello, this is a test input for importance computation."
inputs = tokenizer(text, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}  # **确保输入在 GPU**

# **5. 启用梯度计算**
for param in model.parameters():
    param.requires_grad = True

# **6. 前向传播**
outputs = model(**inputs)
loss = outputs.logits.mean()  # 计算均值损失，避免显存爆炸
loss.backward()  # **计算梯度**

# **7. 计算梯度范数**
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
            grad_norms[key][i] = np.mean([torch.norm(p.to(device), p=2).item() for p in layer_params])  # **确保梯度计算在 GPU**
        else:
            grad_norms[key][i] = 0  # 如果该层没有这个模块，则设为 0

# **8. 归一化梯度范数**
for key in grad_norms:
    max_val = np.max(grad_norms[key])
    if max_val > 0:
        grad_norms[key] /= max_val  # 归一化，确保不同模块之间的数值可比

# **9. 打印数据**
print(grad_norms)

# **10. 保存数据**
np.save("llama_component_grad_norms.npy", grad_norms)
print("Gradient norm importance scores saved to llama_component_grad_norms.npy")
