import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig
from datasets import load_dataset
from torch.utils.data import DataLoader
import math

# 🚀 **VIB 可训练剪枝 Mask**
class VIBMask(nn.Module):
    def __init__(self, size, pruning_ratio=0.5):
        super().__init__()
        self.mu = nn.Parameter(torch.zeros(size))  # 需要梯度
        self.sigma = nn.Parameter(torch.ones(size) * pruning_ratio)  # 需要梯度

    def forward(self, prev_mask=None):
        """ 计算剪枝 mask，同时考虑前面层的 token 依赖性 """
        epsilon = torch.randn_like(self.sigma)
        mask = torch.sigmoid(self.mu + epsilon * self.sigma)

        # 🚀 **让当前层剪枝 Mask 受前面层影响**
        if prev_mask is not None:
            mask = mask * prev_mask

        return mask

    def kl_loss(self):
        """ 计算 KL 散度 loss，让剪枝 Mask 可训练 """
        return -0.5 * torch.mean(1 + self.sigma - self.mu ** 2 - torch.exp(self.sigma))


# 🚀 **Llama Self Attention（支持剪枝）**
class LlamaAttention(nn.Module):
    def __init__(self, config: LlamaConfig, pruning_ratio=0.5):
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

        # 🚀 **GQA 剪枝 Mask**
        self.mask_q = VIBMask(self.num_heads, pruning_ratio)  # Query 头剪枝 Mask
        self.mask_kv = VIBMask(self.num_key_value_heads, pruning_ratio)  # Key/Value 头剪枝 Mask

    def forward(self, hidden_states, attention_mask=None):
        batch_size, seq_length, _ = hidden_states.shape

        # 计算 Q/K/V
        query_states = self.q_proj(hidden_states).view(batch_size, seq_length, self.num_heads, self.head_dim)
        key_states = self.k_proj(hidden_states).view(batch_size, seq_length, self.num_key_value_heads, self.head_dim)
        value_states = self.v_proj(hidden_states).view(batch_size, seq_length, self.num_key_value_heads, self.head_dim)

        # 🚀 **剪枝 Q 头**
        mask_q = self.mask_q()
        query_states = query_states * mask_q

        # 🚀 **剪枝 KV 头**
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

        # 🚀 **返回 KL Loss**
        return attn_output, self.mask_q.kl_loss() + self.mask_kv.kl_loss()


# 🚀 **剪枝 Llama3**
def prune_llama_model(llama_model, pruning_ratios):
    """ 🚀 TVA-Prune 剪枝 Llama3 模型 """
    for i, layer in enumerate(llama_model.model.layers):  
        layer.self_attn.mask_q = VIBMask(layer.self_attn.num_heads, pruning_ratios[i])
        layer.self_attn.mask_kv = VIBMask(layer.self_attn.num_key_value_heads, pruning_ratios[i])
    return llama_model


# 🚀 **加载数据集**
def get_dataloader():
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
    dataset = load_dataset("openwebtext", split="train")

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    tokenized_datasets.set_format(type="torch", columns=["input_ids", "attention_mask"])

    return DataLoader(tokenized_datasets, batch_size=4, shuffle=True)


# 🚀 **训练剪枝 Mask**
def train_mask(model, dataloader, epochs=3, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 🚀 **冻结 Llama3 的所有权重**
    for param in model.parameters():
        param.requires_grad = False

    # 🚀 **只训练剪枝 Mask**
    vib_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(vib_params, lr=lr)

    for epoch in range(epochs):
        model.train()
        total_kl_loss = 0

        for step, batch in enumerate(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            _, kl_loss = model(**batch)  # 🚀 **只计算 KL 损失**

            loss = kl_loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_kl_loss += loss.item()
            if step % 50 == 0:
                print(f"Epoch {epoch+1}, Step {step}, KL Loss: {loss.item()}")

        print(f"Epoch {epoch+1} finished, Avg KL Loss: {total_kl_loss / len(dataloader)}")


if __name__ == "__main__":
    # 🚀 加载 Llama3
    cache_dir = "/root/autodl-tmp/llm_weights"
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B", cache_dir=cache_dir,device_map="auto", torch_dtype=torch.float16)

    # 🚀 设置剪枝比例
    pruning_ratios = [0.7 * i for i in range(32)]

    # 🚀 剪枝模型
    model = prune_llama_model(model, pruning_ratios)

    # 🚀 训练剪枝 Mask
    train_dataloader = get_dataloader()
    train_mask(model, train_dataloader)
