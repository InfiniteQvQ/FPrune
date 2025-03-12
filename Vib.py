import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from datasets import load_dataset
import os

# ✅ 每层的剪枝率
TARGET_SPARSITY_PER_LAYER = [
    0.3, 0.5, 0.6, 0.4, 0.7, 0.5, 0.6, 0.8, 0.5, 0.4, 0.3, 0.5, 0.7, 0.6, 0.8, 0.4,
    0.5, 0.6, 0.7, 0.8, 0.5, 0.4, 0.6, 0.7, 0.5, 0.3, 0.8, 0.7, 0.6, 0.5, 0.9
] * 7  # 32 层，每层不同剪枝率


# ✅ Variational Information Bottleneck (VIB) Mask 层
class VIBLayer(nn.Module):
    def __init__(self, input_dim, target_sparsity):
        super().__init__()
        self.mu = nn.Parameter(torch.randn(input_dim) * 0.01)
        self.log_sigma = nn.Parameter(torch.randn(input_dim) * 0.01)
        self.target_sparsity = target_sparsity

    def forward(self, x):
        std = torch.exp(0.5 * self.log_sigma)
        eps = torch.randn_like(std)
        z = self.mu + eps * std
        mask = torch.sigmoid(z)  # 生成 soft mask

        # 计算阈值，使剪枝率符合 target_sparsity
        sorted_mask, _ = torch.sort(mask)
        threshold = sorted_mask[int(len(mask) * self.target_sparsity)]
        final_mask = (mask > threshold).float()
        return x * final_mask, mask  # 返回剪枝后的张量 & 原始 mask


# ✅ 重新封装 LLaMA，添加 **每层独立** 的 VIB 层
class PrunedLlama(nn.Module):
    def __init__(self, model, target_sparsity_per_layer):
        super().__init__()
        self.model = model
        self.num_layers = len(model.model.layers)

        # 为每一层的 MLP 和 Attention 创建独立的 Mask
        self.mlp_vib_layers = nn.ModuleList([
            VIBLayer(model.model.layers[i].mlp.down_proj.weight.shape[0], target_sparsity_per_layer[i])
            for i in range(self.num_layers)
        ])

        self.attn_vib_layers = nn.ModuleList([
            VIBLayer(model.model.layers[i].self_attn.o_proj.weight.shape[0], target_sparsity_per_layer[i])
            for i in range(self.num_layers)
        ])

    def forward(self, x):
        kl_total = 0  # 统计 KL Loss
        for i, layer in enumerate(self.model.model.layers):
            x, mask_mlp = self.mlp_vib_layers[i](x)
            x, mask_attn = self.attn_vib_layers[i](x)
            x = layer(x)  # 进入 LLaMA 原始结构

            # 计算 KL 损失
            kl_total += kl_loss(self.mlp_vib_layers[i].mu, self.mlp_vib_layers[i].log_sigma)
            kl_total += kl_loss(self.attn_vib_layers[i].mu, self.attn_vib_layers[i].log_sigma)

        return x, kl_total


# ✅ KL 约束 Loss
def kl_loss(mu, log_sigma):
    return -0.5 * torch.sum(1 + log_sigma - mu.pow(2) - log_sigma.exp())


# ✅ 训练函数（多卡支持）
def train_mask(rank, world_size):
    # 1️⃣ 初始化分布式训练
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    # 2️⃣ 加载 LLaMA 模型
    model_name = "pinkmanlove/llama-7b-hf"
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceM4/llama-7b-tokenizer")
    cache_dir = "/root/autodl-tmp/llm_weights"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        torch_dtype=torch.float16
    ).to(rank)

    print(f"🔥 Model has {len(model.model.layers)} layers")
    print(f"🔥 Target sparsity list has {len(TARGET_SPARSITY_PER_LAYER)} values")

    # 3️⃣ 初始化剪枝模型（多卡模式）
    pruned_model = PrunedLlama(model, TARGET_SPARSITY_PER_LAYER).to(rank)
    pruned_model = DDP(pruned_model, device_ids=[rank], output_device=rank)

    # 4️⃣ 加载训练数据
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    train_texts = dataset["text"]
    train_tokens = tokenizer(train_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)

    train_sampler = DistributedSampler(train_tokens["input_ids"], num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(train_tokens["input_ids"], batch_size=4, sampler=train_sampler)

    # 5️⃣ 训练 Mask
    optimizer = optim.Adam(pruned_model.parameters(), lr=5e-4)
    beta = 1e-5

    for epoch in range(5):  # 训练 5 轮
        train_sampler.set_epoch(epoch)  # 设置采样器的 epoch，保证多卡同步
        total_loss = 0
        for inputs in train_loader:
            inputs = inputs.to(rank)

            optimizer.zero_grad()
            outputs, kl_losses = pruned_model(inputs)  # 计算 KL Loss

            loss = outputs.loss + beta * kl_losses
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"[GPU {rank}] Epoch {epoch + 1}, Loss: {total_loss:.4f}")

    # 6️⃣ 保存 Mask（只在主进程保存）
    if rank == 0:
        torch.save(pruned_model.module.state_dict(), "pruned_llama7b.pth")
        print("[GPU 0] 🎯 Mask 已保存！")

    # 7️⃣ 关闭进程组
    dist.destroy_process_group()


# ✅ 运行分布式训练
def main():
    world_size = torch.cuda.device_count()  # 获取 GPU 数量
    print(f"💡 发现 {world_size} 张 GPU，启动多卡训练...")

    mp.spawn(train_mask, args=(world_size,), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
