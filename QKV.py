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

# **创建 Hook 存储注意力权重**
attn_scores = {}

def get_attention_scores_hook(layer_id):
    def hook(module, input, output):
        attn_weights = output[1]  # `attn_probs` 是输出中的第二个
        attn_scores[f"layer_{layer_id}"] = attn_weights.mean(dim=[0, 1, 2]).item()
    return hook

# **给每一层 Self-Attention 机制注册 Hook**
hooks = []
for layer_id, layer in enumerate(model.model.layers):
    hook = layer.self_attn.register_forward_hook(get_attention_scores_hook(layer_id))
    hooks.append(hook)

# **输入一个测试文本**
text = "The quick brown fox jumps over the lazy dog."
inputs = tokenizer(text, return_tensors="pt").to("cuda")

# **执行 Forward Pass**
with torch.no_grad():
    model(**inputs)

# **移除 Hook**
for hook in hooks:
    hook.remove()

# **按注意力分数排序**
sorted_attn = sorted(attn_scores.items(), key=lambda x: -x[1])

# **输出 Top 10 重要的层**
print("Top 10 Most Important Layers by Attention Score:")
for layer, score in sorted_attn[]:
    print(f"{layer}: {score}")
