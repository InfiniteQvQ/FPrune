import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class VIBMask(nn.Module):
    def __init__(self, size, init_sparsity=0.5):
        super(VIBMask, self).__init__()
        self.mu = nn.Parameter(torch.zeros(size))
        self.sigma = nn.Parameter(torch.ones(size) * init_sparsity)
    
    def forward(self):
        epsilon = torch.randn_like(self.sigma)
        mask = torch.sigmoid(self.mu + epsilon * self.sigma)
        return mask

    def binarize(self, threshold=0.5):
        return (torch.sigmoid(self.mu) > threshold).float()


def train_vib_mask(mask, layer, sparsity_target, epochs=100, lr=1e-3):
    optimizer = optim.Adam(mask.parameters(), lr=lr)
    for epoch in range(epochs):
        optimizer.zero_grad()
        vib_mask = mask()
        pruned_layer = layer * vib_mask
        loss_sparsity = torch.abs(vib_mask.mean() - sparsity_target)
        loss_kl = torch.mean(vib_mask * torch.log(vib_mask + 1e-8))
        loss = loss_sparsity + 0.01 * loss_kl  # 0.01 是正则化权重
        loss.backward()
        optimizer.step()
    return mask.binarize()


def prune_llama_layer(llama_model, pruning_ratios):
    for layer_idx, layer in enumerate(llama_model.layers):
        layer_sparsity = pruning_ratios[layer_idx]
        for sub_layer_name in ["gqa", "out", "gate", "up", "down"]:
            if hasattr(layer, sub_layer_name):
                sub_layer = getattr(layer, sub_layer_name)
                mask = VIBMask(sub_layer.weight.shape)
                bin_mask = train_vib_mask(mask, sub_layer.weight, layer_sparsity)
                sub_layer.weight.data *= bin_mask
    return llama_model


def optimize_pruned_model(model):
    for name, param in model.named_parameters():
        if "weight" in name:
            pruned_dim = param.shape[0]
            optimized_dim = (pruned_dim // 128) * 128
            if pruned_dim % 128 != 0:
                optimized_dim += 128
            padding = optimized_dim - pruned_dim
            if padding > 0:
                param.data = torch.cat([param.data, torch.zeros(padding, param.shape[1], device=param.device)], dim=0)
    return model


if __name__ == "__main__":
    from transformers import LlamaForCausalLM, AutoModelForCausalLM
    pruning_ratios = [0.7] * 32
    cache_dir = "/root/autodl-tmp/llm_weights"
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-8B",
        cache_dir=cache_dir,
        device_map="auto", 
        torch_dtype=torch.float16
    )
    model = prune_llama_layer(model, pruning_ratios)
    model = optimize_pruned_model(model)
    model.save_pretrained("pruned_llama3_8b")
