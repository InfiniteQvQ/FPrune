import torch
import numpy as np
from transformers import AutoModelForCausalLM, LlamaTokenizer

# 超参数
ATTN_WEIGHT = 1.5   # Q, K, V 的权重
MLP_WEIGHT = 1.0    # Out, Gate, Up, Down 的权重
S1 = 0.9            # 线性映射下限
S2 = 1.0            # 线性映射上限
EP = 1e-8           # 防止除零

# ==================== 计算 QKV 注意力重要性 ====================
def compute_attention_importance(model, dataloader, device):
    """计算 Transformer Q, K, V 的注意力重要性"""
    num_layers = len(model.model.layers)
    module_importance = {"q_proj": np.zeros(num_layers), "k_proj": np.zeros(num_layers), "v_proj": np.zeros(num_layers)}

    with torch.no_grad():
        for batch in dataloader:
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**inputs, output_attentions=True)
            attentions = outputs.attentions  # shape: (num_layers, batch_size, num_heads, seq_len, seq_len)

            for layer_idx, layer_attention in enumerate(attentions):
                layer_attn_scores = layer_attention.mean(dim=(0, 2, 3)).cpu().numpy()
                module_importance["q_proj"][layer_idx] += layer_attn_scores.mean()
                module_importance["k_proj"][layer_idx] += layer_attn_scores.mean()
                module_importance["v_proj"][layer_idx] += layer_attn_scores.mean()

    for key in module_importance.keys():
        module_importance[key] /= len(dataloader)
    return module_importance

# ==================== 计算 Fisher 信息矩阵 (Out, Gate, Up, Down) ====================
def compute_fisher_information(model, inputs, labels, module_keys, device):
    """计算 Transformer Out, Gate, Up, Down 的 Fisher 信息矩阵"""
    model.zero_grad()
    outputs = model(**inputs)
    loss_fct = torch.nn.CrossEntropyLoss()
    logits = outputs.logits
    loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))  # 计算 loss
    loss.backward()

    fisher_scores = {key: [] for key in module_keys}
    for i, layer in enumerate(model.model.layers):
        module_scores = {}
        for name, param in layer.named_parameters():
            for key in fisher_scores.keys():
                if key in name and param.grad is not None:
                    fisher = torch.mean(param.grad ** 2).item()  # 计算 Fisher 信息
                    module_scores[key] = fisher

        total = sum(module_scores.values()) + EP
        for key in fisher_scores.keys():
            fisher_scores[key].append(module_scores.get(key, 0) / total)
    return fisher_scores

# ==================== 归一化并组合最终分数 ====================
def normalize_and_combine(attn_scores, fisher_scores):
    """对 QKV 和 Out, Gate, Up, Down 进行归一化并组合"""
    num_layers = len(attn_scores["q_proj"])  # Llama 7B 共有 32 层
    importance_scores = []

    for i in range(num_layers):
        # 归一化 Q, K, V
        layer_attn = np.array([attn_scores["q_proj"][i], attn_scores["k_proj"][i], attn_scores["v_proj"][i]])
        min_attn, max_attn = layer_attn.min(), layer_attn.max()
        mapped_attn = ((layer_attn - min_attn) / (max_attn - min_attn + EP)) * (S2 - S1) + S1
        mapped_attn *= ATTN_WEIGHT

        # 归一化 Out, Gate, Up, Down
        layer_fisher = np.array([fisher_scores["o_proj"][i], fisher_scores["gate_proj"][i],
                                 fisher_scores["up_proj"][i], fisher_scores["down_proj"][i]])
        min_fisher, max_fisher = layer_fisher.min(), layer_fisher.max()
        mapped_fisher = ((layer_fisher - min_fisher) / (max_fisher - min_fisher + EP)) * (S2 - S1) + S1
        mapped_fisher *= MLP_WEIGHT

        # 拼接每层的 7 个分数（Q, K, V, O, Gate, Up, Down）
        combined_scores = np.concatenate([mapped_attn, mapped_fisher])
        importance_scores.extend(combined_scores)

    return np.array(importance_scores)

# ==================== 主函数 ====================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 加载 LLaMA 7B 模型
    cache_dir = "/root/autodl-tmp/llm_weights"
    model = AutoModelForCausalLM.from_pretrained(
        "pinkmanlove/llama-7b-hf",
        cache_dir=cache_dir,
        device_map="auto",  # 自动分布到多个 GPU
        torch_dtype=torch.float16
    )

    # 加载 tokenizer
    tokenizer = LlamaTokenizer.from_pretrained("HuggingFaceM4/llama-7b-tokenizer")

    # 生成一个测试样本
    text = "Hello, this is a test input for importance computation."
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 生成 ground-truth（让模型预测下一个 token）
    labels = inputs["input_ids"]

    # 构造 dataloader
    dataloader = [inputs]

    # 计算 Q, K, V 注意力重要性
    attn_scores = compute_attention_importance(model, dataloader, device)

    # 计算 Fisher 信息矩阵
    module_keys = ["o_proj", "gate_proj", "up_proj", "down_proj"]
    fisher_scores = compute_fisher_information(model, inputs, labels, module_keys, device)

    # 归一化并组合最终的 7×32 分数
    importance_scores = normalize_and_combine(attn_scores, fisher_scores)

    # 保存重要性分数
    np.save("importance_scores.npy", importance_scores)
    print(importance_scores)
    print("Importance scores saved to importance_scores.npy")
    print(importance_scores)
