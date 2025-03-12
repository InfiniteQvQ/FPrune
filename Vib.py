import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from accelerate import Accelerator

# -------------------------
# 全局参数设置
# -------------------------
# 假设 LLaMA-7B 模型有 32 层（每层包含 7 个子层），这里每层目标剪枝率为 0.7，
# 意味着该层中 70% 的通道将被剪掉（各子层共享同一隐藏状态 mask）
TARGET_SPARSITY_PER_LAYER = [0.7] * 32

BETA = 1e-5              # KL 正则项权重
NUM_EPOCHS = 5
BATCH_SIZE = 4
LEARNING_RATE = 5e-4
MAX_SEQ_LENGTH = 512

# -------------------------
# 定义 ViB 模块（对隐藏状态通道施加 mask）
# -------------------------
class VIBLayerWrapper(nn.Module):
    def __init__(self, hidden_dim, target_sparsity):
        """
        hidden_dim: 待剪枝通道数（通常为 config.hidden_size）
        target_sparsity: 目标剪枝率，例如 0.7 表示剪掉 70% 的通道
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.mu = nn.Parameter(torch.randn(hidden_dim) * 0.01)
        self.log_sigma = nn.Parameter(torch.randn(hidden_dim) * 0.01)
        self.target_sparsity = target_sparsity

    def forward(self, x):
        """
        x: (batch, seq_len, hidden_dim)
        产生一个 (1, 1, hidden_dim) 的 mask，用于按通道剪枝 x
        """
        std = torch.exp(0.5 * self.log_sigma)
        eps = torch.randn_like(std)
        z = self.mu + eps * std
        mask_prob = torch.sigmoid(z)  # [hidden_dim]
        # 根据排序确定阈值，使得大约 target_sparsity 的比例置 0
        sorted_probs, _ = torch.sort(mask_prob.view(-1))
        cutoff_index = int(len(sorted_probs) * self.target_sparsity)
        threshold = sorted_probs[cutoff_index]
        final_mask = (mask_prob > threshold).float()  # [hidden_dim]
        # 扩展为 (1, 1, hidden_dim) 并移动到 x.device 上
        final_mask = final_mask.view(1, 1, -1).to(x.device)
        x_masked = x * final_mask
        # 计算 KL 损失（未归一化）
        kl = -0.5 * torch.sum(1 + self.log_sigma - self.mu.pow(2) - torch.exp(self.log_sigma))
        return x_masked, kl

# -------------------------
# 定义包装后的 LLaMA 层
# -------------------------
class VIBWrappedLayer(nn.Module):
    def __init__(self, orig_layer, hidden_dim, target_sparsity):
        """
        orig_layer: 原始 LLaMA 层（包含内部 7 个子层）
        hidden_dim: 待剪枝通道数
        target_sparsity: 目标剪枝率
        """
        super().__init__()
        self.orig_layer = orig_layer
        self.vib = VIBLayerWrapper(hidden_dim, target_sparsity)

    def forward(self, x, attention_mask=None, position_ids=None, past_key_value=None, output_attentions=False, use_cache=False):
        # 先对输入 x 施加 ViB mask
        x_masked, kl = self.vib(x)
        # 调用原始层 forward（注意不要传递额外参数）
        out = self.orig_layer(
            x_masked,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache
        )
        # 原始层返回 tuple，第一个元素为隐藏状态输出
        return out[0], kl

# -------------------------
# 定义最终模型包装器，将所有层的 KL 损失累加
# -------------------------
class VIBPrunedLlamaForCausalLM(nn.Module):
    def __init__(self, orig_model):
        """
        orig_model: 原始 AutoModelForCausalLM 模型（其中 model.layers 已经被替换为 VIBWrappedLayer）
        """
        super().__init__()
        self.orig_model = orig_model

    def forward(self, input_ids, attention_mask, labels=None):
        # 获取输入 embedding
        x = self.orig_model.model.embed_tokens(input_ids)
        total_kl = 0.0
        # 顺序执行每一层，累加 KL 损失
        for layer in self.orig_model.model.layers:
            x, kl = layer(
                x,
                attention_mask=attention_mask,
                position_ids=None,
                past_key_value=None,
                output_attentions=False,
                use_cache=False
            )
            total_kl += kl
        if self.orig_model.model.norm is not None:
            x = self.orig_model.model.norm(x)
        logits = self.orig_model.lm_head(x)
        loss = None
        if labels is not None:
            # 右移后计算交叉熵损失
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return {"loss": loss, "logits": logits, "kl_loss": total_kl}

# -------------------------
# 工具函数：扩展 attention mask 到 (batch, 1, seq_len, seq_len)
# -------------------------
def expand_attention_mask(mask):
    # mask: (batch, seq_len) boolean tensor
    # 首先 unsqueeze 为 (batch, 1, 1, seq_len)，再 expand到 (batch, 1, seq_len, seq_len)
    batch_size, seq_len = mask.size()
    mask = mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq_len)
    mask = mask.expand(batch_size, 1, seq_len, seq_len)
    return mask

# -------------------------
# 训练代码（使用 accelerate 管理多 GPU，不手动调用 .to()）
# -------------------------
def train_vib_mask():
    accelerator = Accelerator()

    # 1. 加载原始模型与 tokenizer（device_map="auto" 交由 accelerate 管理）
    model_name = "pinkmanlove/llama-7b-hf"
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceM4/llama-7b-tokenizer")
    cache_dir = "/root/autodl-tmp/llm_weights"  # 根据实际情况设置

    orig_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    # 不要手动调用 .to()，accelerate 会处理

    num_layers = len(orig_model.model.layers)
    if len(TARGET_SPARSITY_PER_LAYER) != num_layers:
        raise ValueError(f"Expected {num_layers} target sparsity values, but got {len(TARGET_SPARSITY_PER_LAYER)}")
    hidden_dim = orig_model.config.hidden_size
    # 替换每一层为 VIBWrappedLayer
    for i in range(num_layers):
        orig_layer = orig_model.model.layers[i]
        vib_layer = VIBWrappedLayer(orig_layer, hidden_dim, TARGET_SPARSITY_PER_LAYER[i])
        orig_model.model.layers[i] = vib_layer

    # 用最终包装器封装模型
    model = VIBPrunedLlamaForCausalLM(orig_model)

    # 2. 加载数据集
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = dataset["text"]
    tokenized = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_SEQ_LENGTH)
    input_ids = tokenized["input_ids"]

    dataloader = DataLoader(input_ids, batch_size=BATCH_SIZE, shuffle=True)

    # 3. 使用 accelerate 准备模型、优化器、数据加载器
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    model.train()
    for epoch in range(NUM_EPOCHS):
        total_loss = 0.0
        for batch in dataloader:
            optimizer.zero_grad()
            # 构造 attention mask：原始 mask 形状 (batch, seq_len) 转换为 (batch, 1, seq_len, seq_len)
            raw_mask = (batch != tokenizer.pad_token_id)
            attn_mask = expand_attention_mask(raw_mask)
            outputs = model(
                input_ids=batch,
                attention_mask=attn_mask,
                labels=batch
            )
            loss = outputs["loss"] + BETA * outputs["kl_loss"]
            accelerator.backward(loss)
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {total_loss:.4f}")

    # 4. 保存模型（accelerate 会自动解包多 GPU 模型）
    save_path = "vib_pruned_llama7b.pth"
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    torch.save(unwrapped_model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    train_vib_mask()
