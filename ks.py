import torch
import numpy as np
from transformers import AutoModelForCausalLM, LlamaTokenizer

# 超参数
ATTN_WEIGHT = 1.5   # Q, K, V 的权重
MLP_WEIGHT = 1.0    # Out, Gate, Up, Down 的权重
EP = 1e-8           # 防止除零

# ==================== 计算 QKV 注意力重要性 ====================
def compute_attention_importance(model, dataloader, device):
    """分别计算 Transformer Q, K, V 的注意力重要性"""
    num_layers = len(model.model.layers)
    module_importance = {"q_proj": np.zeros(num_layers), "k_proj": np.zeros(num_layers), "v_proj": np.zeros(num_layers)}

    with torch.no_grad():
        for batch in dataloader:
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**inputs, output_attentions=True)
            attentions = outputs.attentions  # shape: (num_layers, batch_size, num_heads, seq_len, seq_len)

            for layer_idx in range(num_layers):
                attn = attentions[layer_idx].mean(dim=(0, 3))  # 对 batch 维度和 seq_len 维度取均值
                module_importance["q_proj"][layer_idx] += attn[0].mean().item()  # 取第 1 个 head
                module_importance["k_proj"][layer_idx] += attn[1].mean().item()  # 取第 2 个 head
                module_importance["v_proj"][layer_idx] += attn[2].mean().item()  # 取第 3 个 head

    for key in module_importance.keys():
        module_importance[key] /= len(dataloader)
    return module_importance

# ==================== 计算 Fisher 信息矩阵 (Out, Gate, Up, Down) ====================
def compute_fisher_information(model, inputs, labels, module_keys, device):
    """计算 Transformer Out, Gate, Up, Down 的 Fisher 信息"""
    model.zero_grad()
    outputs = model(**inputs)
    loss_fct = torch.nn.CrossEntropyLoss()
    logits = outputs.logits
    loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))  # 计算 loss
    loss.backward()

    fisher_scores = {key: np.zeros(len(model.model.layers)) for key in module_keys}
    for i, layer in enumerate(model.model.layers):
        for key in module_keys:
            param_list = [p for n, p in layer.named_parameters() if key in n and p.grad is not None]
            if len(param_list) > 0:
                fisher_values = [torch.mean(p.grad ** 2).item() for p in param_list]  # 计算 Fisher 信息
                fisher_scores[key][i] = np.sum(fisher_values)  # 不取均值，直接求和，保持数值差异

    return fisher_scores

# ==================== 组合最终分数 ====================
def combine_scores(attn_scores, fisher_scores):
    """不归一化，直接组合"""
    num_layers = len(attn_scores["q_proj"])
    importance_matrix = np.zeros((num_layers, 7))

    for i in range(num_layers):
        importance_matrix[i, 0] = attn_scores["q_proj"][i] * ATTN_WEIGHT
        importance_matrix[i, 1] = attn_scores["k_proj"][i] * ATTN_WEIGHT
        importance_matrix[i, 2] = attn_scores["v_proj"][i] * ATTN_WEIGHT
        importance_matrix[i, 3] = fisher_scores["o_proj"][i] * MLP_WEIGHT
        importance_matrix[i, 4] = fisher_scores["gate_proj"][i] * MLP_WEIGHT
        importance_matrix[i, 5] = fisher_scores["up_proj"][i] * MLP_WEIGHT
        importance_matrix[i, 6] = fisher_scores["down_proj"][i] * MLP_WEIGHT

    return importance_matrix

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

    # 组合最终重要性分数 (不归一化)
    importance_matrix = combine_scores(attn_scores, fisher_scores)

    # 保存重要性分数
    np.save("importance_scores.npy", importance_matrix)
    print("Importance scores saved to importance_scores.npy")
    print(importance_matrix)
