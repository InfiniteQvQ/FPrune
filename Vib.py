import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

#####################################
# 1) 定义 your InformationBottleneck
#####################################
class InformationBottleneck(nn.Module):
    def __init__(self, size, kl_mult=1.0):
        super().__init__()
        self.size = size
        self.kl_mult = kl_mult
        # 让 mu, sigma 成为可训练参数
        self.mu = nn.Parameter(torch.zeros(size))
        self.sigma = nn.Parameter(torch.ones(size)*0.5)

    def forward(self, x):
        """
        x: [batch, dim], elementwise mask
        """
        eps = torch.randn_like(self.sigma)
        mask = torch.sigmoid(self.mu + eps*self.sigma)
        return x*mask

    def kl_loss(self):
        return -0.5*torch.mean(1 + self.sigma - self.mu.pow(2) - self.sigma.exp())*self.kl_mult

##########################################
# 2) 解析: apply_vib_pruning_ratios
##########################################
def apply_vib_pruning_ratios(model, pruning_ratios):
    layers = model.model.layers
    if len(pruning_ratios)!=len(layers):
        raise ValueError("layer数目不匹配")
    for i, layer in enumerate(layers):
        ratio = pruning_ratios[i]
        if hasattr(layer.self_attn, "ib_1"):
            layer.self_attn.ib_1.sigma.data.fill_(ratio)
            layer.self_attn.ib_1.mu.data.zero_()
        if hasattr(layer.mlp, "ib_1"):
            layer.mlp.ib_1.sigma.data.fill_(ratio)
            layer.mlp.ib_1.mu.data.zero_()
    print("成功为每层应用了剪枝比例。")

##########################################
# 3) freeze_non_mask_params
##########################################
def freeze_non_mask_params(model):
    for name, param in model.named_parameters():
        # 如果 name包含 ib_1，就让它可训练
        if "ib_1" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

##########################################
# 4) 收集 KL Loss
##########################################
def collect_kl_loss(model):
    kl_total = 0.0
    for layer in model.model.layers:
        if hasattr(layer.self_attn, "ib_1"):
            kl_total += layer.self_attn.ib_1.kl_loss()
        if hasattr(layer.mlp, "ib_1"):
            kl_total += layer.mlp.ib_1.kl_loss()
    # 如果 embedding-level mask 也存在
    if getattr(model.model, "hidden_mask", None):
        kl_total += model.model.hidden_mask.kl_loss()
    return kl_total

def main():
    # 1. 加载LLama 3.1-8B
    ckpt_path = "meta-llama/Llama-3.1-8B"
    cache_dir = "/root/autodl-tmp/llm_weights"
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        ckpt_path,
        cache_dir=cache_dir,
        torch_dtype=torch.float16
    ).cuda()
    # 在 config 里启用 vib_layers
    model.config.vib_layers = True
    model.config.use_cache = False

    # 2. 给每层分配剪枝比例
    n_layers = model.config.num_hidden_layers
    pruning_ratios = [0.7]*n_layers
    apply_vib_pruning_ratios(model, pruning_ratios)

    # 3. 冻结非 mask
    freeze_non_mask_params(model)

    # 检查一下是否真的有 param
    param_list = [n for n,p in model.named_parameters() if p.requires_grad]
    if len(param_list)==0:
        raise ValueError("依旧是空列表! 说明 self_attn.ib_1 之类可能没定义 Parameter. 请检查 code.")

    # 4. tokenizer & dataset
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    def tokenize_fn(examples):
        return tokenizer(examples["text"], max_length=128, padding="max_length", truncation=True)
    dataset = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    dataset.set_format(type="torch", columns=["input_ids","attention_mask"])
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # 5. 优化器
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    model.train()

    # 6. 简单训练
    max_steps = 30
    for step, batch in enumerate(dataloader):
        if step>=max_steps:
            break
        input_ids = batch["input_ids"].cuda()
        attention_mask = batch["attention_mask"].cuda()

        # 手动gen position_ids
        bsz, seq_len = input_ids.shape
        pos_ids = attention_mask.long().cumsum(-1)-1
        pos_ids.masked_fill_(attention_mask==0,0)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=pos_ids,
            labels=input_ids
        )
        ce_loss = outputs.loss
        kl = collect_kl_loss(model)
        loss = ce_loss + 0.01*kl

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step%5==0:
            print(f"Step={step}, CE={ce_loss.item():.4f}, KL={kl.item():.4f}, total={loss.item():.4f}")

    print("训练结束, 保存模型")
    model.save_pretrained("/root/autodl-tmp/pruned_llama3_8b")

if __name__=="__main__":
    main()
