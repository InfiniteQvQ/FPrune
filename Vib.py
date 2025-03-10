import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import math

# 🚀 VIB 可训练剪枝 Mask
class VIBMask(nn.Module):
    def __init__(self, size, pruning_ratio=0.5):
        super().__init__()
        self.mu = nn.Parameter(torch.zeros(size))  # 可训练参数
        self.sigma = nn.Parameter(torch.ones(size) * pruning_ratio)

    def forward(self, prev_mask=None):
        epsilon = torch.randn_like(self.sigma)
        mask = torch.sigmoid(self.mu + epsilon * self.sigma)
        if prev_mask is not None:
            mask = mask * prev_mask
        return mask

    def kl_loss(self):
        return -0.5 * torch.mean(1 + self.sigma - self.mu ** 2 - torch.exp(self.sigma))


# 🚀 Llama Self-Attention（支持剪枝）
class LlamaAttention(nn.Module):
    def __init__(self, config, pruning_ratio=0.5):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.hidden_size = config.hidden_size

        # 线性变换
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias)

        # 🚀 GQA 剪枝 Mask：针对 Query 和 Key/Value 分别添加可训练 Mask
        self.mask_q = VIBMask(self.num_heads, pruning_ratio)
        self.mask_kv = VIBMask(self.num_key_value_heads, pruning_ratio)

    def forward(self, hidden_states, attention_mask=None):
        batch_size, seq_length, _ = hidden_states.shape

        # 计算 Q/K/V
        query_states = self.q_proj(hidden_states).view(batch_size, seq_length, self.num_heads, self.head_dim)
        key_states = self.k_proj(hidden_states).view(batch_size, seq_length, self.num_key_value_heads, self.head_dim)
        value_states = self.v_proj(hidden_states).view(batch_size, seq_length, self.num_key_value_heads, self.head_dim)

        # 🚀 剪枝 Query 头
        mask_q = self.mask_q()
        query_states = query_states * mask_q

        # 🚀 剪枝 Key/Value 头
        mask_kv = self.mask_kv()
        key_states = key_states * mask_kv
        value_states = value_states * mask_kv

        # 计算注意力权重
        attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.view(batch_size, seq_length, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        # 返回输出和 Mask 的 KL 损失
        return attn_output, self.mask_q.kl_loss() + self.mask_kv.kl_loss()


# 🚀 Llama MLP（支持剪枝）
class LlamaMLP(nn.Module):
    def __init__(self, config, pruning_ratio=0.5):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size)

        # 为 gate, up, down 添加剪枝 Mask
        self.mask_gate = VIBMask(config.intermediate_size, pruning_ratio)
        self.mask_up = VIBMask(config.intermediate_size, pruning_ratio)
        self.mask_down = VIBMask(config.hidden_size, pruning_ratio)

    def forward(self, hidden_states):
        # 对 gate_proj 和 up_proj 剪枝
        mask_gate = self.mask_gate()
        mask_up = self.mask_up()
        hidden_states = F.silu(self.gate_proj(hidden_states) * mask_gate) * (self.up_proj(hidden_states) * mask_up)
        # 对 down_proj 剪枝
        mask_down = self.mask_down()
        hidden_states = self.down_proj(hidden_states) * mask_down

        # 返回输出和三个 mask 的 KL 损失之和
        return hidden_states, self.mask_gate.kl_loss() + self.mask_up.kl_loss() + self.mask_down.kl_loss()


# 🚀 Llama Decoder Layer（支持剪枝）
class LlamaDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx, pruning_ratio=0.5):
        super().__init__()
        self.attention = LlamaAttention(config, pruning_ratio)
        self.mlp = LlamaMLP(config, pruning_ratio)
        self.norm_1 = nn.LayerNorm(config.hidden_size)
        self.norm_2 = nn.LayerNorm(config.hidden_size)

    def forward(self, hidden_states, attention_mask=None):
        residual = hidden_states
        hidden_states = self.norm_1(hidden_states)
        hidden_states, kl_attn = self.attention(hidden_states, attention_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.norm_2(hidden_states)
        hidden_states, kl_mlp = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        # 累加注意力和 MLP 部分的 KL 损失
        return hidden_states, kl_attn + kl_mlp


# 如果你想从零构建模型，可以定义一个自定义 LlamaModel；
# 但这里我们假设使用预训练的 LlamaForCausalLM，并通过剪枝函数替换其层中的 Mask。
def prune_llama_model(model, pruning_ratios):
    """
    根据每层分配的剪枝比例，替换预训练模型中每层注意力和 MLP 部分的剪枝 Mask。
    注意：预训练 LlamaForCausalLM 的 Decoder 层中注意力模块属性为 `self_attn`，而非我们自定义的 `attention`，
    所以这里需要用 `layer.self_attn` 和 `layer.mlp` 来访问。
    """
    for i, layer in enumerate(model.model.layers):
        pruning_ratio = pruning_ratios[i]
        # 替换注意力中的剪枝 Mask
        layer.self_attn.mask_q = VIBMask(layer.self_attn.num_heads, pruning_ratio)
        layer.self_attn.mask_kv = VIBMask(layer.self_attn.num_key_value_heads, pruning_ratio)
        # 替换 MLP 中的剪枝 Mask
        layer.mlp.mask_gate = VIBMask(layer.mlp.intermediate_size, pruning_ratio)
        layer.mlp.mask_up = VIBMask(layer.mlp.intermediate_size, pruning_ratio)
        layer.mlp.mask_down = VIBMask(layer.mlp.hidden_size, pruning_ratio)
    return model


# 🚀 加载数据集（这里使用 wikitext-2-raw-v1）
def get_dataloader():
    cache_dir = "/root/autodl-tmp"  # 指定缓存目录
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B", cache_dir=cache_dir)
    # 如果没有 pad_token，则用 eos_token 代替
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", cache_dir=cache_dir)

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

    return DataLoader(tokenized_dataset, batch_size=4, shuffle=True)


# 🚀 训练剪枝 Mask（冻结 Llama 权重，仅训练 Mask）
def train_mask(model, dataloader, epochs=3, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # 冻结模型所有权重，只训练 Mask 参数
    for param in model.parameters():
        param.requires_grad = False
    vib_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(vib_params, lr=lr)

    for epoch in range(epochs):
        model.train()
        total_kl_loss = 0
        for step, batch in enumerate(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            # 前向传播只返回 (outputs, kl_loss)
            _, kl_loss = model(**batch)
            loss = kl_loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_kl_loss += loss.item()
            if step % 50 == 0:
                print(f"Epoch {epoch+1}, Step {step}, KL Loss: {loss.item()}")
        print(f"Epoch {epoch+1} finished, Avg KL Loss: {total_kl_loss / len(dataloader)}")


if __name__ == "__main__":
    cache_dir = "/root/autodl-tmp/llm_weights"
    # 🚀 加载预训练 LlamaForCausalLM 模型
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-8B",
        cache_dir=cache_dir,
        device_map="auto",
        torch_dtype=torch.float16
    )

    # 🚀 生成每层剪枝比例，这里示例使用 [0, 0.7*1, 0.7*2, ..., 0.7*31]
    pruning_ratios = [0.7 * i for i in range(model.config.num_hidden_layers)]
    # 如果模型层数不足32，则按实际层数设置

    # 🚀 对预训练模型应用剪枝（替换各层 Mask）
    model = prune_llama_model(model, pruning_ratios)

    # 🚀 加载数据集
    dataloader = get_dataloader()

    # 🚀 训练剪枝 Mask（冻结 Llama 权重，仅训练 Mask 参数）
    train_mask(model, dataloader)

    # 训练完成后可以保存模型
    model.save_pretrained("/root/autodl-tmp/pruned_llama3_8b")
