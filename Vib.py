import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import math
from accelerate import Accelerator

# ------------------------------
# VIBMask：可训练剪枝 Mask
# ------------------------------
class VIBMask(nn.Module):
    def __init__(self, size, pruning_ratio=0.5):
        super().__init__()
        self.mu = nn.Parameter(torch.zeros(size))
        self.sigma = nn.Parameter(torch.ones(size) * pruning_ratio)

    def forward(self, prev_mask=None):
        epsilon = torch.randn_like(self.sigma)
        mask = torch.sigmoid(self.mu + epsilon * self.sigma)
        if prev_mask is not None:
            mask = mask * prev_mask
        return mask

    def kl_loss(self):
        return -0.5 * torch.mean(
            1 + self.sigma - self.mu ** 2 - torch.exp(self.sigma)
        )

# ------------------------------
# LlamaAttention（支持 VIB 剪枝）
# ------------------------------
class LlamaAttention(nn.Module):
    def __init__(self, config, pruning_ratio=0.5):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
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

        # Q, K, V
        query_states = self.q_proj(hidden_states).view(batch_size, seq_length, self.num_heads, self.head_dim)
        key_states = self.k_proj(hidden_states).view(batch_size, seq_length, self.num_key_value_heads, self.head_dim)
        value_states = self.v_proj(hidden_states).view(batch_size, seq_length, self.num_key_value_heads, self.head_dim)

        # VIB mask
        mask_q = self.mask_q()
        query_states = query_states * mask_q
        mask_kv = self.mask_kv()
        key_states = key_states * mask_kv
        value_states = value_states * mask_kv

        # 计算注意力
        attn_weights = torch.matmul(
            query_states, key_states.transpose(-2, -1)
        ) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.view(batch_size, seq_length, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output, self.mask_q.kl_loss() + self.mask_kv.kl_loss()

# ------------------------------
# LlamaMLP（支持 VIB 剪枝）
# ------------------------------
class LlamaMLP(nn.Module):
    def __init__(self, config, pruning_ratio=0.5):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size)

        self.mask_gate = VIBMask(config.intermediate_size, pruning_ratio)
        self.mask_up = VIBMask(config.intermediate_size, pruning_ratio)
        self.mask_down = VIBMask(config.hidden_size, pruning_ratio)

    def forward(self, hidden_states):
        mask_gate = self.mask_gate()
        mask_up = self.mask_up()

        hidden_states = F.silu(
            self.gate_proj(hidden_states) * mask_gate
        ) * (self.up_proj(hidden_states) * mask_up)

        mask_down = self.mask_down()
        hidden_states = self.down_proj(hidden_states) * mask_down

        return hidden_states, (
            self.mask_gate.kl_loss()
            + self.mask_up.kl_loss()
            + self.mask_down.kl_loss()
        )

# ------------------------------
# LlamaDecoderLayer（支持 VIB 剪枝）
# ------------------------------
class LlamaDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx, pruning_ratio=0.5):
        super().__init__()
        self.self_attn = LlamaAttention(config, pruning_ratio)
        self.mlp = LlamaMLP(config, pruning_ratio)
        self.norm_1 = nn.LayerNorm(config.hidden_size)
        self.norm_2 = nn.LayerNorm(config.hidden_size)

    def forward(self, hidden_states, attention_mask=None):
        # Self-Attn
        residual = hidden_states
        hidden_states = self.norm_1(hidden_states)
        hidden_states, kl_attn = self.self_attn(hidden_states, attention_mask)
        hidden_states = residual + hidden_states

        # MLP
        residual = hidden_states
        hidden_states = self.norm_2(hidden_states)
        hidden_states, kl_mlp = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, kl_attn + kl_mlp

# ------------------------------
# 自定义 forward：遍历 decoder 层并累加 KL 损失
# ------------------------------
def custom_llama_forward(self, input_ids, attention_mask=None, **kwargs):
    """
    覆盖模型的 forward，用于遍历 decoder 层并累加 VIB KL Loss。
    另外，手动生成 position_ids，避免 NoneType 报错。
    """
    batch_size, seq_length = input_ids.shape

    # 🔑 这里手动生成 position_ids，避免 Llama 在内部因 NoneType 报错
    if attention_mask is not None:
        # 常见做法：对非 padding 位置计数
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 0)
    else:
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=input_ids.device
        ).unsqueeze(0).expand(batch_size, -1)

    hidden_states = self.embed_tokens(input_ids)

    kl_total = 0
    for layer in self.layers:
        hidden_states, kl_loss = layer(hidden_states, attention_mask)
        kl_total += kl_loss

    hidden_states = self.norm(hidden_states)
    # LlamaForCausalLM 可能需要返回 (hidden_states, kl_total)
    return hidden_states, kl_total

def override_forward(model):
    """
    将自定义 forward 绑定到 model.model.forward，覆盖默认 Llama forward。
    """
    model.model.forward = custom_llama_forward.__get__(model.model, type(model.model))

# ------------------------------
# 替换预训练模型中每层的剪枝 Mask
# ------------------------------
def prune_llama_model(model, pruning_ratios):
    for i, layer in enumerate(model.model.layers):
        pruning_ratio = pruning_ratios[i]
        # 注意力部分
        layer.self_attn.mask_q = VIBMask(layer.self_attn.num_heads, pruning_ratio)
        layer.self_attn.mask_kv = VIBMask(layer.self_attn.num_key_value_heads, pruning_ratio)
        # MLP部分
        layer.mlp.mask_gate = VIBMask(layer.mlp.intermediate_size, pruning_ratio)
        layer.mlp.mask_up = VIBMask(layer.mlp.intermediate_size, pruning_ratio)
        layer.mlp.mask_down = VIBMask(layer.mlp.hidden_size, pruning_ratio)
    return model

# ------------------------------
# 加载数据集（使用 wikitext-2-raw-v1）
# ------------------------------
def get_dataloader():
    cache_dir = "/root/autodl-tmp"
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B", cache_dir=cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", cache_dir=cache_dir)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=512
        )

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    return DataLoader(tokenized_dataset, batch_size=4, shuffle=True)

# ------------------------------
# 冻结非 Mask 参数，仅训练 VIBMask 参数
# ------------------------------
def freeze_non_mask_params(model):
    for name, param in model.named_parameters():
        if "mask" not in name:
            param.requires_grad = False

# ------------------------------
# 训练剪枝 Mask（使用 Accelerate 管理多 GPU）
# ------------------------------
def train_mask(model, dataloader, epochs=3, lr=1e-4):
    accelerator = Accelerator()

    # 1. 冻结非 mask 参数
    freeze_non_mask_params(model)

    # 2. 只训练 VIBMask
    vib_params = [p for p in model.parameters() if p.requires_grad]
    if not vib_params:
        raise ValueError(
            "optimizer got an empty parameter list; check that VIBMask parameters are not frozen."
        )

    optimizer = torch.optim.AdamW(vib_params, lr=lr)

    # 3. 用 accelerator.prepare(...) 管理多 GPU
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    for epoch in range(epochs):
        model.train()
        total_kl_loss = 0

        for step, batch in enumerate(dataloader):
            # 🔑 手动生成 position_ids，拼接到 batch 中
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]

            batch_size, seq_length = input_ids.shape

            # 常见方法：对非 padding 位置计数
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 0)

            # 在 forward 时，会将它用于 Llama 内部
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids  # 传递 position_ids，避免 NoneType 报错
            )

            # 自定义 forward 返回 (hidden_states, kl_loss)
            _, kl_loss = outputs
            loss = kl_loss

            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            total_kl_loss += loss.item()
            if step % 50 == 0:
                print(f"Epoch {epoch+1}, Step {step}, KL Loss: {loss.item()}")

        print(f"Epoch {epoch+1} finished, Avg KL Loss: {total_kl_loss / len(dataloader)}")

if __name__ == "__main__":
    cache_dir = "/root/autodl-tmp/llm_weights"

    # 1. 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-8B",
        cache_dir=cache_dir,
        torch_dtype=torch.float16
    )

    # 2. 覆盖 forward
    override_forward(model)

    # 3. 对每层应用剪枝
    pruning_ratios = [0.7 * i for i in range(model.config.num_hidden_layers)]
    model = prune_llama_model(model, pruning_ratios)

    # 4. 加载数据集
    dataloader = get_dataloader()

    # 5. 训练剪枝 Mask
    train_mask(model, dataloader)

    # 6. 保存模型
    model.save_pretrained("/root/autodl-tmp/pruned_llama3_8b")
