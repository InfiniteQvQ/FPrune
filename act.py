import torch
import numpy as np
from transformers import AutoModelForCausalLM

def count_parameters(model):
    """Iterates through each layer of the LLaMA model and counts the total parameters."""
    layer_params = {}
    
    if hasattr(model, 'model'):
        layers = model.model.layers  # Extract transformer layers
    else:
        raise ValueError("Invalid model structure. No layers found.")
    
    for i, layer in enumerate(layers):
        layer_total = 0
        sublayer_params = {}
        
        for name, param in layer.named_parameters():
            num_params = param.numel()
            layer_total += num_params
            sublayer_params[name] = num_params
        
        layer_params[f"Layer {i}"] = {
            "Total Params": layer_total,
            "Breakdown": sublayer_params
        }
    
    return layer_params

if __name__ == "__main__":
    cache_dir = "/root/autodl-tmp/llm_weights"
    model = AutoModelForCausalLM.from_pretrained(
        "pinkmanlove/llama-7b-hf",
        cache_dir=cache_dir,
        device_map="auto",
        torch_dtype=torch.float16
    )

    
    param_info = count_parameters(model)
    
    # Display results
    for layer, details in param_info.items():
        print(f"{layer}: {details['Total Params']:,} parameters")
        for sublayer, count in details["Breakdown"].items():
            print(f"  {sublayer}: {count:,} parameters")
        print("-")
