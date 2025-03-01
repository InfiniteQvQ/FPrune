import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

cache_dir = "/root/autodl-tmp/llm_weights"
model = AutoModelForCausalLM.from_pretrained(
    "pinkmanlove/llama-7b-hf",
    cache_dir=cache_dir,
    device_map="auto",  # 让 Hugging Face 自动分配多个 GPU
    torch_dtype=torch.float16
)

tokenizer_name = "HuggingFaceM4/llama-7b-tokenizer"
tokenizer = LlamaTokenizer.from_pretrained(tokenizer_name)

# Function to extract activations per transformer block
# Function to extract activations per transformer block
def get_activations(model, inputs):
    activations = {layer: {"Q": [], "K": [], "V": [], "OUT": [], "GATE": [], "UP": [], "DOWN": []} for layer in range(32)}
    
    def hook_fn(module, input, output, key, layer_id):
        activations[layer_id][key].append(output.detach().cpu().numpy())
    
    for name, module in model.named_modules():
        for layer_id in range(32):
            if f"model.layers.{layer_id}.self_attn.q_proj" in name:
                module.register_forward_hook(lambda mod, inp, out, l=layer_id: hook_fn(mod, inp, out, "Q", l))
            elif f"model.layers.{layer_id}.self_attn.k_proj" in name:
                module.register_forward_hook(lambda mod, inp, out, l=layer_id: hook_fn(mod, inp, out, "K", l))
            elif f"model.layers.{layer_id}.self_attn.v_proj" in name:
                module.register_forward_hook(lambda mod, inp, out, l=layer_id: hook_fn(mod, inp, out, "V", l))
            elif f"model.layers.{layer_id}.self_attn.o_proj" in name:
                module.register_forward_hook(lambda mod, inp, out, l=layer_id: hook_fn(mod, inp, out, "OUT", l))
            elif f"model.layers.{layer_id}.mlp.gate_proj" in name:
                module.register_forward_hook(lambda mod, inp, out, l=layer_id: hook_fn(mod, inp, out, "GATE", l))
            elif f"model.layers.{layer_id}.mlp.up_proj" in name:
                module.register_forward_hook(lambda mod, inp, out, l=layer_id: hook_fn(mod, inp, out, "UP", l))
            elif f"model.layers.{layer_id}.mlp.down_proj" in name:
                module.register_forward_hook(lambda mod, inp, out, l=layer_id: hook_fn(mod, inp, out, "DOWN", l))
    
    with torch.no_grad():
        model(**inputs)
    
    return activations

# Generate synthetic input
def generate_synthetic_input(tokenizer, max_length=128):
    text = "This is a test input for LLaMA model analysis. " * 5
    inputs = tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True)
    return inputs

# Run inference and get activations
inputs = generate_synthetic_input(tokenizer)
activations = get_activations(model, inputs)

# Compute variance for each transformer block
def compute_variance(activations):
    variance_results = {layer: {} for layer in range(32)}
    for layer in range(32):
        for key in activations[layer]:
            if activations[layer][key]:
                stacked_activations = np.concatenate(activations[layer][key], axis=0)
                stacked_activations = np.nan_to_num(stacked_activations, nan=0.0, posinf=1e10, neginf=-1e10)
                variance_results[layer][key] = np.var(stacked_activations)
    return variance_results

variance_results = compute_variance(activations)

# Convert to structured data for visualization
data = []
for layer, values in variance_results.items():
    for key, var in values.items():
        data.append([layer, key, var])


# Sort and print results
print("LLaMA 7B Layer-wise Importance Based on Activation Variance:")
for layer in range(32):
    sorted_vars = sorted(variance_results[layer].items(), key=lambda x: x[1], reverse=True)
    print(f"Layer {layer}: {sorted_vars}")
