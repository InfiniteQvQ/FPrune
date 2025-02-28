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
text = "Hello, this is a test input for pruning."
inputs = tokenizer(text, return_tensors="pt")

# Move inputs to the same device as model layers
inputs = {key: value.to(model.device) for key, value in inputs.items()}

# Forward pass with loss computation
outputs = model.forward(**inputs, labels=inputs["input_ids"])  # Ensure inputs align with model device
loss = outputs.loss
loss.backward()  # Compute gradients

# Store Fisher importance scores
fisher_scores = []

# Iterate through Transformer layers in LLaMA
for layer_idx, layer in enumerate(model.model.layers):
    target_layers = [
        layer.self_attn.q_proj,  # Q projection
        layer.self_attn.k_proj,  # K projection
        layer.self_attn.v_proj,  # V projection
        layer.self_attn.o_proj,  # Output projection
        layer.mlp.gate_proj,     # Gated MLP projection
        layer.mlp.up_proj,       # Up projection
        layer.mlp.down_proj,     # Down projection
    ]

    for module in target_layers:
        if module.weight.grad is not None:
            fisher_score = torch.abs(module.weight * module.weight.grad).sum().item()
            fisher_scores.append(fisher_score)

# Convert to NumPy array
fisher_scores = np.array(fisher_scores)

# Save Fisher scores to a file (optional)
np.save("llama_7b_fisher.npy", fisher_scores)

# Print sample results
print("Fisher Scores (1D Array):", fisher_scores[:10])  # Show first 10 values
