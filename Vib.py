import torch
import torch.nn.functional as F
import torch.nn as nn
import math
from transformers import LlamaPreTrainedModel, LlamaConfig, AutoModelForCausalLM
from torch.utils.data import DataLoader, Dataset
import tqdm

class VIBMask(nn.Module):
    def __init__(self, size, pruning_ratio=0.5):
        super().__init__()
        self.mu = nn.Parameter(torch.zeros(size))  # 需要梯度
        self.sigma = nn.Parameter(torch.ones(size) * pruning_ratio)  # 需要梯度

    def forward(self, prev_mask=None):
        epsilon = torch.randn_like(self.sigma)
        mask = torch.sigmoid(self.mu + epsilon * self.sigma)

        if prev_mask is not None:
            mask = mask * prev_mask

        return mask

    def kl_loss(self):
        return -0.5 * torch.mean(1 + self.sigma - self.mu ** 2 - torch.exp(self.sigma))  


class LlamaAttention(nn.Module):
    def __init__(self, config: LlamaConfig, pruning_ratio=0.5):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.hidden_size = config.hidden_size

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias)

        self.mask_q = VIBMask(self.num_heads, pruning_ratio)  
        self.mask_kv = VIBMask(self.num_key_value_heads, pruning_ratio)

    def forward(self, hidden_states, attention_mask=None):
        batch_size, seq_length, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states).view(batch_size, seq_length, self.num_heads, self.head_dim)
        key_states = self.k_proj(hidden_states).view(batch_size, seq_length, self.num_key_value_heads, self.head_dim)
        value_states = self.v_proj(hidden_states).view(batch_size, seq_length, self.num_key_value_heads, self.head_dim)

        mask_q = self.mask_q()
        query_states = query_states * mask_q

        mask_kv = self.mask_kv()
        key_states = key_states * mask_kv
        value_states = value_states * mask_kv

        attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.view(batch_size, seq_length, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output, self.mask_q.kl_loss() + self.mask_kv.kl_loss()


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
        hidden_states, kl_loss = self.attention(hidden_states, attention_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.ffn(F.gelu(self.norm_2(hidden_states)))
        hidden_states = residual + self.output(hidden_states)

        return hidden_states, kl_loss


class LlamaModel(LlamaPreTrainedModel):
    def __init__(self, config, pruning_ratios):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)

        self.layers = nn.ModuleList([
            LlamaDecoderLayer(config, i, pruning_ratios[i])
            for i in range(config.num_hidden_layers)
        ])
        self.norm = nn.LayerNorm(config.hidden_size)

    def forward(self, input_ids, attention_mask=None):
        hidden_states = self.embed_tokens(input_ids)
        kl_total = 0
        for layer in self.layers:
            hidden_states, kl_loss = layer(hidden_states, attention_mask)
            kl_total += kl_loss
        hidden_states = self.norm(hidden_states)
        return hidden_states, kl_total


def prune_llama_model(llama_model, pruning_ratios):
    for i, layer in enumerate(llama_model.model.layers):  
        layer.attention.mask_q = VIBMask(layer.attention.num_heads, pruning_ratios[i])
        layer.attention.mask_kv = VIBMask(layer.attention.num_key_value_heads, pruning_ratios[i])

    return llama_model


if __name__ == "__main__":
    cache_dir = "/root/autodl-tmp/llm_weights"

    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-8B",
        cache_dir=cache_dir,
        device_map="auto", 
        torch_dtype=torch.float16
    )

    pruning_ratios = [0.7 * (i / 32) for i in range(32)]  

    model = prune_llama_model(model, pruning_ratios)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    dataloader = DataLoader(...)  # 加载你的数据集

    for epoch in range(3):
        loop = tqdm.tqdm(dataloader, desc=f"Epoch {epoch+1}")
        for batch in loop:
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to("cuda")
            attention_mask = batch["attention_mask"].to("cuda")
            labels = batch["labels"].to("cuda")

            logits, kl_loss = model(input_ids, attention_mask)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1)) + kl_loss

            loss.backward()
            optimizer.step()

            loop.set_postfix(loss=loss.item())

    model.save_pretrained("/root/autodl-tmp/pruned_llama3_8b")
