import torch
import numpy as np
from transformers import AutoModelForCausalLM, LlamaTokenizer

# Load LLaMA 7B model
cache_dir = "/root/autodl-tmp/llm_weights"
model = AutoModelForCausalLM.from_pretrained(
    "pinkmanlove/llama-7b-hf",
    cache_dir=cache_dir,
    device_map="auto",  # Auto-distribute across GPUs
    torch_dtype=torch.float16
)

# Load tokenizer (ensure consistency with the model)
tokenizer = LlamaTokenizer.from_pretrained("HuggingFaceM4/llama-7b-tokenizer")

# Set model to training mode
model.train()

# Prepare input data
text_list = [
    "Hello, this is a test input for pruning.",
    "The quick brown fox jumps over the lazy dog.",
    "Artificial Intelligence is transforming the world.",
    "In a distant future, humans and AI coexist in harmony.",
    "OpenAI's ChatGPT demonstrates impressive reasoning capabilities.",
    "Large Language Models have revolutionized natural language processing."
]

# Tokenize multiple texts at once
inputs = tokenizer(text_list, return_tensors="pt", padding=True, truncation=True)

# Move inputs to the same device as model layers
inputs = {key: value.to(model.device) for key, value in inputs.items()}

# Forward pass with loss computation
outputs = model.forward(**inputs, labels=inputs["input_ids"])  # Ensure inputs align with model device
loss = outputs.loss
loss.backward()  # Compute gradients

# Fisher importance storage
fisher_scores = []
fisher_per_layer = {
    "q_proj": [],
    "k_proj": [],
    "v_proj": [],
    "o_proj": [],
    "gate_proj": [],
    "up_proj": [],
    "down_proj": [],
}

# Iterate through Transformer layers in LLaMA
for layer_idx, layer in enumerate(model.model.layers):
    target_layers = {
        "q_proj": layer.self_attn.q_proj,  # Q projection
        "k_proj": layer.self_attn.k_proj,  # K projection
        "v_proj": layer.self_attn.v_proj,  # V projection
        "o_proj": layer.self_attn.o_proj,  # Output projection
        "gate_proj": layer.mlp.gate_proj,  # Gated MLP projection
        "up_proj": layer.mlp.up_proj,      # Up projection
        "down_proj": layer.mlp.down_proj,  # Down projection
    }

    for name, module in target_layers.items():
        if module.weight.grad is not None:
            fisher_score = torch.abs(module.weight * module.weight.grad).sum().item()
            fisher_scores.append(fisher_score)
            fisher_per_layer[name].append(fisher_score)

# Convert to NumPy array
fisher_scores = np.array(fisher_scores)

# Compute average Fisher score for each module type
average_fisher_scores = {key: np.mean(values) for key, values in fisher_per_layer.items()}

# Save Fisher scores to a file (optional)
np.save("llama_7b_fisher.npy", fisher_scores)

# Print results
print("Fisher Scores (1D Array, first 10 values):", fisher_scores)  # Show first 10 values
print("\nAverage Fisher Scores per Layer Type:")
for key, value in average_fisher_scores.items():
    print(f"{key}: {value:.4f}")
