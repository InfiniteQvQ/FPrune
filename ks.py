import torch
import numpy as np
from transformers import AutoModelForCausalLM, LlamaTokenizer

# 超参数
ATTN_WEIGHT = 1.5   # Q, K, V 的权重
MLP_WEIGHT = 1.0    # Out, Gate, Up, Down 的权重
S1 = 0.8          # 线性映射下限
S2 = 1.2           # 线性映射上限
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

            for layer_idx, layer_attention in enumerate(attentions):
                # 计算每个 head 的均值
                attn_scores = layer_attention.mean(dim=(0, 2, 3)).cpu().numpy()
                module_importance["q_proj"][layer_idx] += attn_scores[0]  # 取 Q 投影的均值
                module_importance["k_proj"][layer_idx] += attn_scores[1]  # 取 K 投影的均值
                module_importance["v_proj"][layer_idx] += attn_scores[2]  # 取 V 投影的均值

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
                fisher_values = [torch.mean(p.grad ** 2).item() for p in param_list]
                fisher_scores[key][i] = np.mean(fisher_values)  # 计算该层的 Fisher 信息

    return fisher_scores

# ==================== 归一化并组合最终分数 ====================
def normalize_importance_scores(scores_dict):
    """对 QKV 和 Out, Gate, Up, Down 进行归一化"""
    for key in scores_dict:
        arr = scores_dict[key]
        min_val, max_val = arr.min(), arr.max()
        scores_dict[key] = ((arr - min_val) / (max_val - min_val + EP)) * (S2 - S1) + S1

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

    # 归一化所有得分
    normalize_importance_scores(attn_scores)
    normalize_importance_scores(fisher_scores)

    # 组合 32 层的 7 个数值，形成 (32×7) 的 numpy 数组
    num_layers = len(model.model.layers)
    importance_matrix = np.zeros((num_layers, 7))

    for i in range(num_layers):
        importance_matrix[i, 0] = attn_scores["q_proj"][i] * ATTN_WEIGHT
        importance_matrix[i, 1] = attn_scores["k_proj"][i] * ATTN_WEIGHT
        importance_matrix[i, 2] = attn_scores["v_proj"][i] * ATTN_WEIGHT
        importance_matrix[i, 3] = fisher_scores["o_proj"][i] * MLP_WEIGHT
        importance_matrix[i, 4] = fisher_scores["gate_proj"][i] * MLP_WEIGHT
        importance_matrix[i, 5] = fisher_scores["up_proj"][i] * MLP_WEIGHT
        importance_matrix[i, 6] = fisher_scores["down_proj"][i] * MLP_WEIGHT

    # 保存重要性分数
    np.save("importance_scores.npy", importance_matrix)
    print(importance_matrix)
    print("Importance scores saved to importance_scores.npy")
    print(importance_matrix)
