import torch
import torch.nn.functional as F
import torch.nn as nn
import math
from transformers import LlamaPreTrainedModel, LlamaConfig

class VIBMask(nn.Module):
    def __init__(self, size, pruning_ratio=0.5):
        super().__init__()
        self.mu = nn.Parameter(torch.zeros(size))
        self.sigma = nn.Parameter(torch.ones(size) * pruning_ratio)

    def forward(self, prev_mask=None):
        """ 计算剪枝 mask，同时考虑前面层的 token 依赖性 """
        epsilon = torch.randn_like(self.sigma)
        mask = torch.sigmoid(self.mu + epsilon * self.sigma)

        # 🚀 **让当前层剪枝 Mask 受前面层影响**
        if prev_mask is not None:
            mask = mask * prev_mask

        return mask

    def binarize(self, threshold=0.5):
        return (torch.sigmoid(self.mu) > threshold).float()


class LlamaAttention(nn.Module):
    def __init__(self, config: LlamaConfig, pruning_ratio=0.5):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads  # GQA 共享比例
        self.head_dim = config.hidden_size // self.num_heads
        self.hidden_size = config.hidden_size

        # 线性变换
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias)

        # GQA 剪枝 Mask
        self.mask_q = VIBMask(self.num_heads, pruning_ratio)  # Query 头剪枝 Mask
        self.mask_kv = VIBMask(self.num_key_value_heads, pruning_ratio)  # Key/Value 头剪枝 Mask

    def forward(self, hidden_states, attention_mask=None):
        batch_size, seq_length, _ = hidden_states.shape

        # 计算 Q/K/V
        query_states = self.q_proj(hidden_states).view(batch_size, seq_length, self.num_heads, self.head_dim)
        key_states = self.k_proj(hidden_states).view(batch_size, seq_length, self.num_key_value_heads, self.head_dim)
        value_states = self.v_proj(hidden_states).view(batch_size, seq_length, self.num_key_value_heads, self.head_dim)

        # 🚀 **剪枝 Q 头**
        query_states = query_states * self.mask_q()

        # 🚀 **剪枝 KV 头**
        key_states = key_states * self.mask_kv()
        value_states = value_states * self.mask_kv()

        # 由于 Query 共享 Key/Value，GQA 需要对 KV 进行 repeat
        key_states = key_states.repeat(1, 1, self.num_key_value_groups, 1)
        value_states = value_states.repeat(1, 1, self.num_key_value_groups, 1)

        # **计算注意力权重**
        attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.view(batch_size, seq_length, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        return attn_output


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx, pruning_ratio=0.5):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.attention = LlamaAttention(config, pruning_ratio)
        self.ffn = nn.Linear(config.hidden_size, config.intermediate_size)
        self.output = nn.Linear(config.intermediate_size, config.hidden_size)
        self.norm_1 = nn.LayerNorm(config.hidden_size)
        self.norm_2 = nn.LayerNorm(config.hidden_size)

    def forward(self, hidden_states, attention_mask=None):
        residual = hidden_states
        hidden_states = self.norm_1(hidden_states)
        hidden_states = self.attention(hidden_states, attention_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.norm_2(hidden_states)
        hidden_states = self.ffn(hidden_states)
        hidden_states = F.gelu(hidden_states)
        hidden_states = self.output(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class LlamaModel(LlamaPreTrainedModel):
    def __init__(self, config, pruning_ratios):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)

        # 每层都传入不同的 pruning_ratio
        self.layers = nn.ModuleList([
            LlamaDecoderLayer(config, i, pruning_ratios[i])
            for i in range(config.num_hidden_layers)
        ])
        self.norm = nn.LayerNorm(config.hidden_size)

    def forward(self, input_ids, attention_mask=None):
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        hidden_states = self.norm(hidden_states)
        return hidden_states


def prune_llama_model(llama_model, pruning_ratios):
    """ 🚀 TVA-Prune 剪枝 Llama 模型 """
    
    for i, layer in enumerate(llama_model.layers):
        layer.attention.mask_q = VIBMask(layer.attention.num_heads, pruning_ratios[i])
        layer.attention.mask_kv = VIBMask(layer.attention.num_key_value_heads, pruning_ratios[i])

    return llama_model


if __name__ == "__main__":
    from transformers import AutoModelForCausalLM

    cache_dir = "/root/autodl-tmp/llm_weights"

    # 1️⃣ 加载 Llama
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-8B",
        cache_dir=cache_dir,
        device_map="auto", 
        torch_dtype=torch.float16
    )

    # 2️⃣ 生成每层不同的剪枝比例
    pruning_ratios = [0.7 * i for i in range(32)]  # 第一层 40%，后续每层递增 1%

    # 3️⃣ TVA-Prune 剪枝
    model = prune_llama_model(model, pruning_ratios)

    # 4️⃣ 保存剪枝后的模型
    model.save_pretrained("/root/autodl-tmp/pruned_llama3_8b")
