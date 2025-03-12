import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# -------------------------
# 全局参数设置
# -------------------------
# 假设 LLaMA-7B 模型有 32 层（实际层数请根据模型检查），这里每层的目标剪枝率设为 0.7（即剪掉 70%）
TARGET_SPARSITY_PER_LAYER = [0.7] * 32

BETA = 1e-5              # KL 正则项权重
NUM_EPOCHS = 5
BATCH_SIZE = 4
LEARNING_RATE = 5e-4
MAX_SEQ_LENGTH = 512

# -------------------------
# 定义 VIB 模块
# -------------------------
class VIBLayer(nn.Module):
    def __init__(self, input_dim, target_sparsity):
        """
        input_dim: 待剪枝向量的维度（例如 mlp.down_proj.weight 的行数或 attn.o_proj.weight 的行数）
        target_sparsity: 目标剪枝率（例如 0.7 表示剪掉 70% 的权重）
        """
        super().__init__()
        self.mu = nn.Parameter(torch.randn(input_dim) * 0.01)
        self.log_sigma = nn.Parameter(torch.randn(input_dim) * 0.01)
        self.target_sparsity = target_sparsity

    def forward(self, x):
        """
        x: (batch_size, seq_len, hidden_dim)
        对 x 的最后一维施加一个 mask
        """
        std = torch.exp(0.5 * self.log_sigma)
        eps = torch.randn_like(std)
        z = self.mu + eps * std
        mask_prob = torch.sigmoid(z)
        # 根据排序得到阈值，使得大约 target_sparsity 的比例置 0
        sorted_mask, _ = torch.sort(mask_prob.view(-1))
        cutoff_index = int(len(sorted_mask) * self.target_sparsity)
        threshold = sorted_mask[cutoff_index]
        final_mask = (mask_prob > threshold).float()
        # 将 mask 扩展为 (1, 1, hidden_dim)，便于广播
        final_mask = final_mask.view(1, 1, -1)
        return x * final_mask, mask_prob

# -------------------------
# 定义封装后的 Pruned LLaMA 模型（多 GPU 使用 DataParallel）
# -------------------------
class PrunedLlamaForCausalLM(nn.Module):
    def __init__(self, orig_model, target_sparsity_per_layer):
        """
        orig_model: 原始由 Hugging Face 加载的 AutoModelForCausalLM 模型
        target_sparsity_per_layer: 长度与模型层数相同的列表，表示每一层的目标剪枝率
        """
        super().__init__()
        self.orig_model = orig_model  # 包含 model.model.layers、lm_head、norm、embed_tokens 等
        self.num_layers = len(orig_model.model.layers)
        # 为每一层创建两个 VIB 层（分别作用于 MLP 和 Attention 的输出）
        self.mlp_vib_layers = nn.ModuleList([
            VIBLayer(orig_model.model.layers[i].mlp.down_proj.weight.shape[0],
                     target_sparsity_per_layer[i])
            for i in range(self.num_layers)
        ])
        self.attn_vib_layers = nn.ModuleList([
            VIBLayer(orig_model.model.layers[i].self_attn.o_proj.weight.shape[0],
                     target_sparsity_per_layer[i])
            for i in range(self.num_layers)
        ])

    def forward(self, input_ids, attention_mask, labels=None):
        # 获取输入 embedding
        inputs_embeds = self.orig_model.model.embed_tokens(input_ids)
        x = inputs_embeds

        kl_total = 0.0  # 累计所有 VIB 层的 KL 损失
        # 遍历每一层，先对输入施加 MLP 与 Attention 的 VIB mask，再调用原始 LLaMA 层
        for i, layer in enumerate(self.orig_model.model.layers):
            x, _ = self.mlp_vib_layers[i](x)
            x, _ = self.attn_vib_layers[i](x)
            layer_outputs = layer(
                x,
                attention_mask=attention_mask,
                position_ids=None,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
                hidden_mask=None
            )
            x = layer_outputs[0]
            kl_total += kl_loss(self.mlp_vib_layers[i].mu, self.mlp_vib_layers[i].log_sigma)
            kl_total += kl_loss(self.attn_vib_layers[i].mu, self.attn_vib_layers[i].log_sigma)

        # 经过最后的归一化层与 lm_head 得到 logits
        if self.orig_model.model.norm is not None:
            x = self.orig_model.model.norm(x)
        logits = self.orig_model.lm_head(x)

        loss = None
        if labels is not None:
            # 右移计算交叉熵损失
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return {"loss": loss, "logits": logits, "kl_loss": kl_total}

# KL 损失函数
def kl_loss(mu, log_sigma):
    return -0.5 * torch.sum(1 + log_sigma - mu.pow(2) - log_sigma.exp())

# -------------------------
# 训练代码（多 GPU DataParallel 版）
# -------------------------
def train_vib_mask():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 加载原始模型和 tokenizer
    model_name = "pinkmanlove/llama-7b-hf"
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceM4/llama-7b-tokenizer")
    cache_dir = "/root/autodl-tmp/llm_weights"  # 根据实际情况设置

    orig_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    orig_model.to(device)
    print(f"原始模型层数: {len(orig_model.model.layers)}")
    print(f"目标剪枝率列表长度: {len(TARGET_SPARSITY_PER_LAYER)}")

    # 2. 封装模型，加入每层独立的 VIB Mask，并用 DataParallel 包装
    pruned_model = PrunedLlamaForCausalLM(orig_model, TARGET_SPARSITY_PER_LAYER)
    pruned_model = nn.DataParallel(pruned_model)
    pruned_model.to(device)
    pruned_model.train()

    optimizer = optim.Adam(pruned_model.parameters(), lr=LEARNING_RATE)

    # 3. 加载数据集
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = dataset["text"]
    tokenized = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_SEQ_LENGTH)
    input_ids = tokenized["input_ids"]

    dataloader = DataLoader(input_ids, batch_size=BATCH_SIZE, shuffle=True)

    # 4. 训练循环
    for epoch in range(NUM_EPOCHS):
        total_loss = 0.0
        for batch in dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()
            # 使用输入作为 labels（自回归语言模型任务）
            outputs = pruned_model(input_ids=batch, attention_mask=(batch != tokenizer.pad_token_id), labels=batch)
            if outputs["loss"] is None:
                continue
            loss = outputs["loss"] + BETA * outputs["kl_loss"]
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {total_loss:.4f}")

    # 5. 保存模型（或仅保存 mask 参数）
    save_path = "pruned_llama7b_vib.pth"
    torch.save(pruned_model.module.state_dict(), save_path)
    print(f"模型（包含 VIB mask）已保存到 {save_path}")

if __name__ == "__main__":
    train_vib_mask()
