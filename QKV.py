import torch
from transformers import AutoModelForCausalLM, LlamaTokenizer

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

# **存储注意力分数**
attn_scores = {}

def get_attention_scores_hook(layer_id):
    def hook(module, input, output):
        if isinstance(output, tuple) and len(output) > 1:
            attn_weights = output[1]  # 检查 `attn_probs`
            if attn_weights is not None:
                attn_scores[f"layer_{layer_id}"] = attn_weights.mean(dim=[0, 1, 2]).item()
    return hook

# **注册 Hook**
hooks = []
for layer_id, layer in enumerate(model.model.layers):
    hook = layer.self_attn.register_forward_hook(get_attention_scores_hook(layer_id))
    hooks.append(hook)

# **测试输入**
text = "The quick brown fox jumps over the lazy dog."
inputs = tokenizer(text, return_tensors="pt").to("cuda")

# **执行 Forward Pass，确保返回注意力权重**
with torch.no_grad():
    model(**inputs, output_attentions=True)  # 🔹 关键修正：确保输出 attn_probs

# **移除 Hook**
for hook in hooks:
    hook.remove()

# **排序并输出**
sorted_attn = sorted(attn_scores.items(), key=lambda x: -x[1])
print("Top 10 Most Important Layers by Attention Score:")
for layer, score in sorted_attn[:]:
    print(f"{layer}: {score}")
