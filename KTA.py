import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer

def compute_KTA(H, T):
    """Compute Kernel Target Alignment (KTA)"""
    H = H.astype(np.float64)  # Ensure float64 to prevent overflow
    H /= np.linalg.norm(H, axis=1, keepdims=True) + 1e-8  # Normalize and prevent div-by-zero
    H = np.clip(H, -1e3, 1e3)  # Clip values to avoid overflow

    K = H @ H.T  # Compute Kernel matrix K = H * H^T
    T = T @ T.T  # Compute target kernel matrix

    K_norm = np.sqrt(np.sum(K**2)) + 1e-8  # Prevent div-by-zero
    T_norm = np.sqrt(np.sum(T**2)) + 1e-8

    inner_product = np.sum(K * T)
    return inner_product / (K_norm * T_norm)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "meta-llama/Llama-3.2-3b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(
    model_name, cache_dir="./llm_weights",
    output_hidden_states=True, torch_dtype=torch.float16, device_map="auto"
).to(device)

# Get the number of layers correctly
num_layers = model.config.num_hidden_layers  # Fix: remove len()

# Input text
text = ["LLaMA 3.2 3B Kernel Target Alignment computation."]
inputs = tokenizer(text, return_tensors="pt")
inputs = {key: val.to(device) for key, val in inputs.items()}

num_runs = 5  # Number of runs to compute the mean
seq_len = inputs["input_ids"].shape[1]

kta_results = {layer: [] for layer in range(num_layers)}

for run in range(num_runs):
    with torch.no_grad():
        outputs = model(**inputs)
    hidden_states = outputs.hidden_states  # Get all layers

    # Generate consistent labels
    np.random.seed(42)  
    labels = np.random.choice([-1, 1], size=(seq_len, 1))
    T = labels @ labels.T

    for layer_idx in range(num_layers):
        H = hidden_states[layer_idx][0].cpu().numpy()  # Extract batch 0
        kta_value = compute_KTA(H, T)
        kta_results[layer_idx].append(kta_value)

# Compute mean and std deviation
kta_final = {layer: (np.mean(values), np.std(values)) for layer, values in kta_results.items()}

# Save results
import json
with open("KTA_results.json", "w") as f:
    json.dump(kta_final, f, indent=4)

# Print results
for layer, (mean_kta, std_kta) in kta_final.items():
    print(f"Layer {layer} KTA: {mean_kta:.4f} ± {std_kta:.4f}")
