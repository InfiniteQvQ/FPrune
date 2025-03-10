import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

# ===============================
# 1) 定义可训练剪枝 Mask (VIBMask) 并给 Q/K/V/gate/up/down 等设置
# ===============================
class VIBMask(nn.Module):
    def __init__(self, size, pruning_ratio=0.5):
        super().__init__()
        # mu, sigma 代表 Variational Bottleneck 的高斯分布参数
        # pruning_ratio 可作为初始 sigma 或 mu
        self.mu = nn.Parameter(torch.zeros(size))
        self.sigma = nn.Parameter(torch.ones(size) * pruning_ratio)

    def forward(self):
        eps = torch.randn_like(self.sigma)
        mask = torch.sigmoid(self.mu + eps * self.sigma)
        return mask

    def kl_loss(self):
        """VIB 风格的 KL 散度"""
        return -0.5 * torch.mean(1 + self.sigma - self.mu.pow(2) - torch.exp(self.sigma))

def apply_vib_pruning_ratios(model, pruning_ratios):
    """
    给 LLaMA 的 decoder layers 分配剪枝比例.
    我们假设这里面 self_attn 和 mlp 中的 ib_1(InformationBottleneck) 就是我们要写入的地方。
    也可以改写 Q/K/V/gate/up/down 的 mask sigma/ mu.
    
    :param model: AutoModelForCausalLM 返回的 LLaMA (带 vib_layers=True)
    :param pruning_ratios: list, 长度=模型层数
    """
    # 访问 model.model.layers
    layers = model.model.layers
    if len(pruning_ratios) != len(layers):
        raise ValueError(f"pruning_ratios数目{len(pruning_ratios)} 与 LLaMA层数 {len(layers)} 不匹配")

    for i, layer in enumerate(layers):
        ratio = pruning_ratios[i]
        # 假设在 layer.self_attn 里有: ib_1(InformationBottleneck)
        # 在 layer.mlp 里有: ib_1(InformationBottleneck)
        # 你需要把 ratio 写进这些 ib_1. 例如:
        if hasattr(layer.self_attn, "ib_1"):
            # 简单地把 sigma 填成 ratio
            layer.self_attn.ib_1.post_z_mu.data.fill_(ratio)  # 仅举例
            layer.self_attn.ib_1.post_z_logD.data.fill_(-9.0) # 仅举例
        if hasattr(layer.mlp, "ib_1"):
            layer.mlp.ib_1.post_z_mu.data.fill_(ratio)
            layer.mlp.ib_1.post_z_logD.data.fill_(-9.0)

    print("成功为每层应用了剪枝比例")

def freeze_non_mask_params(model):
    """冻结除包含 mask/ib_1 之外的参数"""
    for name, param in model.named_parameters():
        if "ib_1" not in name and "mask" not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

def collect_vib_kl_loss(model):
    """遍历 layers, 收集 self_attn.ib_1, mlp.ib_1 的 kl_loss"""
    kl_total = 0.0
    for layer in model.model.layers:
        if hasattr(layer.self_attn, "ib_1"):
            kl_total += layer.self_attn.ib_1.kl_loss()
        if hasattr(layer.mlp, "ib_1"):
            kl_total += layer.mlp.ib_1.kl_loss()
    # 同理 embed-level mask (model.model.hidden_mask) 也可以加
    if getattr(model.model, "hidden_mask", None) is not None:
        kl_total += model.model.hidden_mask.kl_loss()
    return kl_total


def main():
    # ================================
    # 2) 加载 LLaMA 3.1-8B (需要注意大显存)
    # ================================
    ckpt_path = "meta-llama/Llama-3.1-8B"
    print(f"Loading model {ckpt_path} ...")
    cache_dir = "/root/autodl-tmp/llm_weights"
    # 需保证在 huggingface 上能加载, 或已下载在本地
    model = AutoModelForCausalLM.from_pretrained(
        ckpt_path,
        cache_dir=cache_dir,
        torch_dtype=torch.float16,
    ).cuda()
    # 强制 vib_layers=True (你可以改 LlamaConfig 后再 from_pretrained)
    model.config.vib_layers = True
    model.config.use_cache = False  # 避免 cache
    # 3) 每层分配剪枝比例
    n_layers = model.config.num_hidden_layers
    # 我们给每层设 0.5, 你可自定义
    pruning_ratios = [0.5]*n_layers
    apply_vib_pruning_ratios(model, pruning_ratios)

    # 4) 冻结非 mask
    freeze_non_mask_params(model)

    # ================================
    # 加载 tokenizer, dataset
    # ================================
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    def tokenize_fn(examples):
        return tokenizer(examples["text"], max_length=256, padding="max_length", truncation=True)
    dataset = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # ================================
    # 训练
    # ================================
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    model.train()

    steps = 100  # 演示100步
    for step, batch in enumerate(dataloader):
        if step>steps:
            break
        input_ids = batch["input_ids"].cuda()
        attention_mask = batch["attention_mask"].cuda()
        # position_ids
        bsz, seq_len = input_ids.shape
        position_ids = attention_mask.long().cumsum(-1)-1
        position_ids.masked_fill_(attention_mask==0, 0)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=input_ids, # 语言模型常用
            use_cache=False
        )
        # outputs.loss 是 CE
        ce_loss = outputs.loss
        # 收集 KL
        kl = collect_vib_kl_loss(model)
        loss = ce_loss + 0.01*kl
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step%10==0:
            print(f"[Step {step}] CE={ce_loss.item():.4f}, KL={kl.item():.4f}, total={loss.item():.4f}")

    print("训练结束，保存模型")
    model.save_pretrained("/root/autodl-tmp/pruned_llama3_8b")

if __name__=="__main__":
    main()
