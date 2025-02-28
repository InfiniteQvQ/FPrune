import torch
import torch.nn as nn
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM

# **加载 LLaMA 7B 模型**
cache_dir = "/root/autodl-tmp/llm_weights"
model = AutoModelForCausalLM.from_pretrained(
    "pinkmanlove/llama-7b-hf",
    cache_dir=cache_dir,
    device_map="auto",
    torch_dtype=torch.float16
)
tokenizer_name = "HuggingFaceM4/llama-7b-tokenizer"
tokenizer = LlamaTokenizer.from_pretrained(tokenizer_name)

device = "cuda" if torch.cuda.is_available() else "cpu"


# **测试输入**
sample_text = "The quick brown fox jumps over the lazy dog."
inputs = tokenizer(sample_text, return_tensors="pt").to(device)

# **前向传播，获取基准输出**
with torch.no_grad():
    base_output = model(**inputs).logits

# **定义扰动幅度**
delta = 1e-3

# **存储影响分数**
influence_scores = {}

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
        param.weight.data += delta

        # **前向传播**
        with torch.no_grad():
            perturbed_output = model(**inputs).logits

        # **计算影响分数**
        influence_score = torch.norm(base_output - perturbed_output).item() / delta
        influence_scores[f"Layer {layer_idx} - {name}"] = influence_score

        # **恢复原始权重**
        param.weight.data.copy_(original_weight)

# **显示影响分数**
sorted_scores = sorted(influence_scores.items(), key=lambda x: x[1], reverse=True)

# **打印前 10 个最重要的层**
print("\n🔥 Top-10 Most Important Layers 🔥")
for name, score in sorted_scores[:10]:
    print(f"{name}: {score:.6f}")

# **打印前 10 个最不重要的层（适合剪枝）**
print("\n❄️ Top-10 Least Important Layers (Pruning Candidates) ❄️")
for name, score in sorted_scores[-10:]:
    print(f"{name}: {score:.6f}")
