import torch
from transformers import AutoModelForCausalLM, LlamaTokenizer

# **加载 LLaMA 7B 模型**
cache_dir = "/root/autodl-tmp/llm_weights"
model = AutoModelForCausalLM.from_pretrained(
    "pinkmanlove/llama-7b-hf",
    cache_dir=cache_dir,
    device_map="auto",  # 让 Hugging Face 自动分配多个 GPU
    torch_dtype=torch.float16
)

tokenizer_name = "HuggingFaceM4/llama-7b-tokenizer"
tokenizer = LlamaTokenizer.from_pretrained(tokenizer_name)

# **测试输入**
sample_text = "The quick brown fox jumps over the lazy dog."
inputs = tokenizer(sample_text, return_tensors="pt")

# ✅ **修正关键代码**
device = model.device  # 获取模型所在设备
inputs = {k: v.to(device) for k, v in inputs.items()}  # 把输入数据转到 `model.device`

# **前向传播，获取基准输出**
with torch.no_grad():
    base_output = model(**inputs).logits  # 现在不会报错了

print("Model inference completed successfully!")

# **定义扰动幅度**
delta = 1e-3

# **存储影响分数**
influence_scores = []

# **计算每一层的影响分数**
for layer_idx, layer in enumerate(model.model.layers):
    print(f"Processing Layer {layer_idx}...")

    # 需要计算的权重
    components = {
        "Q": layer.self_attn.q_proj,
        "K": layer.self_attn.k_proj,
        "V": layer.self_attn.v_proj,
        "Output": layer.self_attn.o_proj,
        "Gate": layer.mlp.gate_proj,
        "Up": layer.mlp.up_proj,
        "Down": layer.mlp.down_proj,
    }

    for name, param in components.items():
        original_weight = param.weight.data.clone()

        # **扰动权重**
        param.weight.data.copy_(original_weight + delta)

        # **前向传播**
        with torch.no_grad():
            perturbed_output = model(**inputs).logits

        # **计算影响分数**
        influence_score = torch.norm(base_output - perturbed_output).item() / delta
        influence_scores.append(influence_score)

        # **恢复原始权重**
        param.weight.data.copy_(original_weight)

# **转换为 1D Tensor**
influence_tensor = torch.tensor(influence_scores, dtype=torch.float32, device=device)

# **打印影响分数**
print("\n🔥 Influence Scores for Each Component 🔥")
print(influence_tensor)


