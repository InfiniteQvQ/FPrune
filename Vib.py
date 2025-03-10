import torch
import torch.nn as nn
from transformers import LlamaForCausalLM, LlamaConfig, AutoModelForCausalLM

class InformationBottleneck(nn.Module):
    def __init__(self, dim, prune_ratio=0.5):
        super().__init__()
        self.dim = dim
        self.prune_ratio = prune_ratio  # 该层的目标剪枝比例
        self.epsilon = 1e-8
        
        self.mu = nn.Parameter(torch.ones(dim))  # 重要性参数
        self.logD = nn.Parameter(torch.zeros(dim))  # 计算 logα
    
    def get_mask(self):
        """ 根据目标剪枝比例计算 mask """
        logalpha = self.logD - torch.log(self.mu.pow(2) + self.epsilon)
        num_pruned = int(self.dim * self.prune_ratio)
        
        sorted_indices = torch.argsort(logalpha)
        mask = torch.ones_like(logalpha)
        mask[sorted_indices[:num_pruned]] = 0  # 剪枝
        
        return mask.float()

    def forward(self, x):
        mask = self.get_mask()
        return x * mask


def compute_pruning_ratios(layer_config, target_sparsity):
    total_params = sum(layer_config.values())
    prune_target = total_params * target_sparsity
    
    pruning_ratios = {}
    for key, param_count in layer_config.items():
        pruning_ratios[key] = prune_target * (param_count / total_params) / param_count
    
    return pruning_ratios


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        target_sparsity = 0.7  # 目标剪枝比例
        
        layer_config = {
            "q_proj": config.hidden_size * config.hidden_size,
            "k_proj": config.hidden_size * config.hidden_size,
            "v_proj": config.hidden_size * config.hidden_size,
            "o_proj": config.hidden_size * config.hidden_size,
            "gate_proj": config.hidden_size * config.ffn_dim,
            "up_proj": config.hidden_size * config.ffn_dim,
            "down_proj": config.ffn_dim * config.hidden_size
        }
        
        pruning_ratios = compute_pruning_ratios(layer_config, target_sparsity)

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size)
        
        self.gate_proj = nn.Linear(config.hidden_size, config.ffn_dim)
        self.up_proj = nn.Linear(config.hidden_size, config.ffn_dim)
        self.down_proj = nn.Linear(config.ffn_dim, config.hidden_size)
        
        self.ib_q = InformationBottleneck(config.hidden_size, pruning_ratios["q_proj"])
        self.ib_k = InformationBottleneck(config.hidden_size, pruning_ratios["k_proj"])
        self.ib_v = InformationBottleneck(config.hidden_size, pruning_ratios["v_proj"])
        self.ib_o = InformationBottleneck(config.hidden_size, pruning_ratios["o_proj"])
        
        self.ib_gate = InformationBottleneck(config.ffn_dim, pruning_ratios["gate_proj"])
        self.ib_up = InformationBottleneck(config.ffn_dim, pruning_ratios["up_proj"])
        self.ib_down = InformationBottleneck(config.hidden_size, pruning_ratios["down_proj"])

    def forward(self, x):
        q = self.ib_q(self.q_proj(x))
        k = self.ib_k(self.k_proj(x))
        v = self.ib_v(self.v_proj(x))
        o = self.ib_o(self.o_proj(x))
        
        x = self.ib_gate(self.gate_proj(x))
        x = self.ib_up(self.up_proj(x))
        x = self.ib_down(self.down_proj(x))

        return x


class PrunedLlamaModel(nn.Module):
    def __init__(self, model_name="meta-llama/Meta-Llama-3-8B"):
        super().__init__()
        cache_dir = "/root/autodl-tmp/llm_weights"
        self.config = LlamaConfig.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.1-8B",
            cache_dir=cache_dir,
            device_map="auto",  # 让 Hugging Face 自动分配多个 GPU
            torch_dtype=torch.float16
        )

        self.model.model.layers = nn.ModuleList([
            LlamaDecoderLayer(self.config, i) for i in range(self.config.num_hidden_layers)
        ])

    def forward(self, input_ids, attention_mask=None):
        return self.model(input_ids, attention_mask=attention_mask)


if __name__ == "__main__":
    model = PrunedLlamaModel()
    print("Pruned LLaMA model loaded successfully.")