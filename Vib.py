# train_vib_llama.py

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from accelerate import Accelerator
from transformers import AutoTokenizer
import os

from my_llama_vib import (
    VIBLlamaForCausalLM,
    apply_vib_pruning_ratios,
    freeze_non_mask_params,
)

def main():
    accelerator = Accelerator()
    ckpt_path = "meta-llama/Llama-3.1-8B"
    print(f"Loading LLaMA 3.1-8B from {ckpt_path}...")
    cache_dir = "/root/autodl-tmp/llm_weights"
    model = VIBLlamaForCausalLM.from_pretrained(
        ckpt_path,
        cache_dir=cache_dir,
        torch_dtype=torch.float16,
        device_map="auto",   # 让 accelerate + HF handle 2 GPU
    )
    # 强制 vib_layers
    model.config.vib_layers = True
    model.config.att_mul = 1.0   # 你自定义
    model.config.inter_mul = 1.0 # 你自定义

    # 1) 分配每层剪枝率
    n_layer = model.config.num_hidden_layers
    # 例子：全部=0.5
    pruning_ratios = [0.5]*n_layer
    apply_vib_pruning_ratios(model, pruning_ratios)

    # 2) 冻结非mask
    freeze_non_mask_params(model)

    # 检查一下可训练参数
    vib_params = [n for n,p in model.named_parameters() if p.requires_grad]
    if len(vib_params)==0:
        raise ValueError("空列表，检查 `ib_` / `hidden_mask`等命名是否成功！")

    # 3) tokenizer + dataset
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    def tokenize_fn(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=256)
    ds = ds.map(tokenize_fn, batched=True, remove_columns=["text"])
    ds.set_format(type="torch", columns=["input_ids","attention_mask"])
    dl = DataLoader(ds, batch_size=1, shuffle=True)

    # 4) 优化器
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)

    # 5) accelerate prepare
    model, optimizer, dl = accelerator.prepare(model, optimizer, dl)
    model.train()

    # 6) 训练loop
    steps=50
    for step, batch in enumerate(dl):
        if step>=steps:
            break
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        # 生成 position_ids
        pos_ids = attention_mask.cumsum(-1)-1
        pos_ids.masked_fill_(attention_mask==0, 0)

        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=pos_ids,
            labels=input_ids
        )
        ce_loss = out.loss
        kl = model.get_vib_kl_loss()
        loss = ce_loss + 0.01*kl

        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()

        if step%10==0:
            accelerator.print(f"step={step}, CE={ce_loss.item():.4f}, KL={kl.item():.4f}, total={loss.item():.4f}")

    # 7) 保存
    accelerator.print("训练结束，保存...")
    unwrapped = accelerator.unwrap_model(model)
    unwrapped.save_pretrained("pruned_llama_3.1_8b_vib", safe_serialization=False)

if __name__=="__main__":
    main()
