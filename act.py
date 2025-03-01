import torch
from torch import nn
from datasets import load_dataset
from transformers import LlamaForCausalLM, LlamaTokenizer,  AutoModelForCausalLM
# 如果是Llama2, 可以用 LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", ...)，
# 如果是老版7B, 可以用 decapoda 的 "decapoda-research/llama-7b-hf" (需要自己转换).
# 这里只是个示例, 你需要改成你自己的模型名称或本地路径

# ---- 1. 加载模型与tokenizer ----
cache_dir = "/root/autodl-tmp/llm_weights"
model = AutoModelForCausalLM.from_pretrained(
    "pinkmanlove/llama-7b-hf",
    cache_dir=cache_dir,
    device_map="auto",  # 让 Hugging Face 自动分配多个 GPU
    torch_dtype=torch.float16
)

tokenizer_name = "HuggingFaceM4/llama-7b-tokenizer"
tokenizer = LlamaTokenizer.from_pretrained(tokenizer_name)

# device_map="auto" 让transformers自动将部分layer放在GPU/CPU, 具体硬件需求较大,仅示例

model.eval()

# ---- 2. 准备一小份评测数据 (wikitext-2) ----
#    我们只取前1-2个batch来做演示
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")  
# wikitext-2 test集有2k多行

# 取前面几条做示例
few_samples = dataset.select(range(4))  # 只取4条, 演示足矣(越多越耗时)
text_list = few_samples["text"]
# 拼成一个批量
# 这里是极简做法:把若干行拼起来, 做一个batch, 并假设不超过最大长度
# 实际你可能得写collate_fn/自动分割等

joined_text = "\n".join(text_list)
inputs = tokenizer(joined_text, return_tensors="pt", truncation=True, max_length=512)
# targets与inputs shift一位 (自回归LM任务) - huggingface内部会自动做, 也可自己实现

input_ids = inputs["input_ids"].cuda()  # 如果device有cuda, 否则改成 .to(model.device)
attention_mask = inputs["attention_mask"].cuda()

# ---- 3. 准备一个语言模型的损失函数(自回归) ----
#   HuggingFace 的 LlamaForCausalLM.forward 支持 labels 即可自动返回loss
labels = input_ids.clone()

with torch.no_grad():
    # 先测下生成
    out_text = model.generate(input_ids, max_new_tokens=20)
    print("Sample generate:", tokenizer.decode(out_text[0]))

# 我们需要计算 Hessian 的对角线 => 需要create_graph=True, 不能no_grad
def forward_loss(model, input_ids, attention_mask, labels):
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels
    )
    # outputs.loss: CrossEntropyLoss of next-token-pred
    return outputs.loss

# ---- 4. Hessian对角近似 (Pearlmutter) ----
def hessian_diag(model, loss):
    """
    使用Pearlmutter trick近似Hessian对角线.
    返回:
      g (flatten): dLoss/dParams
      h (flatten): diag(H)
    """
    grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)

    # flatten grads
    flat_grads = []
    params_list = []
    for g, p in zip(grads, model.parameters()):
        flat_grads.append(g.reshape(-1))
        params_list.append(p)
    full_g = torch.cat(flat_grads)  # [N]

    # Hessian diag ~ hvp / g, 当 v = g
    # hvp = d/dp( sum_i g_i^2 ) => 2H*g?
    # 这里做简化 => hvp = grad( (full_g * full_g).sum(), params )
    # (full_g*full_g).sum() = g^T g
    # 计算
    hvp_tensors = torch.autograd.grad(
        (full_g * full_g).sum(),  # g^T g
        params_list,
        create_graph=False
    )

    # hvp / g => diag(H)
    flat_h = []
    offset = 0
    for g_i, hvp_i in zip(flat_grads, hvp_tensors):
        hvp_i_flat = hvp_i.reshape(-1)
        # 避免除0
        h_i = torch.zeros_like(hvp_i_flat)
        mask = (g_i.abs() > 1e-15)
        h_i[mask] = hvp_i_flat[mask] / g_i[mask]
        flat_h.append(h_i)

    full_h = torch.cat(flat_h)  # diag(H)
    return full_g.detach(), full_h.detach()

# ---- 5. 运行一次前向+后向, 得到 (g,h) 并计算子模块得分 ----
def compute_importance_scores(model, input_ids, attention_mask, labels):
    model.zero_grad()
    loss = forward_loss(model, input_ids, attention_mask, labels)
    g, h = hessian_diag(model, loss)

    # 二阶score: score_i = 0.5 * w_i^2 * h_i
    # 先flatten所有param
    offset = 0
    scores_each_param = torch.zeros_like(g)
    param_info = []
    for p in model.parameters():
        n = p.numel()
        w_flat = p.detach().reshape(-1)
        # 计算 local_score
        local_g = g[offset: offset+n]
        local_h = h[offset: offset+n]
        local_score = 0.5 * (w_flat**2) * local_h
        scores_each_param[offset: offset+n] = local_score
        param_info.append((p, offset, offset+n))
        offset += n

    # 按 Q/K/V/OUT/GATE/UP/DOWN 分类
    module_scores = {"Q":0,"K":0,"V":0,"OUT":0,"GATE":0,"UP":0,"DOWN":0}
    offset = 0
    for (name, param) in model.named_parameters():
        n = param.numel()
        seg = scores_each_param[offset : offset+n]
        s_val = seg.sum().item()
        offset += n

        # 识别Q/K/V/GATE/UP/DOWN/OUT
        # 以LLaMA的param命名为例, 可能是: 
        # - "model.layers.X.self_attn.q_proj.weight"
        # - "model.layers.X.mlp.gate_proj.weight"
        # - "model.layers.X.mlp.up_proj.weight"
        # - "model.layers.X.mlp.down_proj.weight"
        # - "model.layers.X.self_attn.o_proj.weight" (out)
        # - ...
        # 这里做简单关键字判断:
        lname = name.lower()
        if "q_proj" in lname:
            module_scores["Q"] += s_val
        elif "k_proj" in lname:
            module_scores["K"] += s_val
        elif "v_proj" in lname:
            module_scores["V"] += s_val
        elif "gate_proj" in lname:
            module_scores["GATE"] += s_val
        elif "up_proj" in lname:
            module_scores["UP"] += s_val
        elif "down_proj" in lname:
            module_scores["DOWN"] += s_val
        elif "o_proj" in lname or "out_proj" in lname:
            module_scores["OUT"] += s_val
        else:
            pass

    return module_scores

# 实际执行:
loss = forward_loss(model, input_ids, attention_mask, labels)
print("Loss on our tiny wikitext-2 sample: ", loss.item())

# 由于二阶需要 create_graph=True, 我们再单独跑:
model.zero_grad()
scores_dict = compute_importance_scores(model, input_ids, attention_mask, labels)

# 排序看谁更大
sorted_items = sorted(scores_dict.items(), key=lambda x: x[1], reverse=True)
print("=== Module importance (Hessian diag) ===")
for k,v in sorted_items:
    print(f"{k} => {v:.4e}")

# 这样就可以看到Gate/Up/Down等各自的二阶敏感度分数
