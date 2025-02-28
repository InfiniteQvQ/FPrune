import torch
import numpy as np
from transformers import AutoModelForCausalLM, LlamaTokenizer

# 加载 LLaMA 7B 模型
cache_dir = "/root/autodl-tmp/llm_weights"
model = AutoModelForCausalLM.from_pretrained(
    "pinkmanlove/llama-7b-hf",
    cache_dir=cache_dir,
    device_map="auto",
    torch_dtype=torch.float16
)

tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# 选择 device
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.train()  # 需要开启训练模式以计算梯度

# 准备输入数据（示例文本）
text = "Hello, this is a test input for pruning."
inputs = tokenizer(text, return_tensors="pt").to(device)

# 计算前向传播和损失
outputs = model(**inputs, labels=inputs["input_ids"])
loss = outputs.loss
loss.backward()  # 计算梯度

# 存储 Fisher 重要性的数组
fisher_scores = []

# 遍历所有 Transformer 层
for layer_idx, layer in enumerate(model.model.layers):
    target_layers = [
        layer.self_attn.q_proj,  # Q
        layer.self_attn.k_proj,  # K
        layer.self_attn.v_proj,  # V
        layer.self_attn.o_proj,  # Output gate
        layer.mlp.up_proj,  # Up
        layer.mlp.down_proj,  # Down
    ]

    for module in target_layers:
        if module.weight.grad is not None:
            fisher_score = torch.abs(module.weight * module.weight.grad).sum().item()  # 计算 Fisher 重要性
            fisher_scores.append(fisher_score)

# 转换为 NumPy 数组
fisher_scores = np.array(fisher_scores)

# 保存 Fisher 数值到文件（可选）
np.save("llama_7b_fisher.npy", fisher_scores)

# 打印部分结果
print("Fisher Scores (1D Array):", fisher_scores[:10])  # 仅显示前 10 个
