import torch
import numpy as np
from transformers import AutoModelForCausalLM, LlamaTokenizer

# 设置超参数
ATTN_WEIGHT = 1.5  # Q, K, V 的权重
MLP_WEIGHT = 1.0  # O, Gate, Up, Down 的权重
S1 = 0.9         # 线性映射下限
S2 = 1.0         # 线性映射上限
EP = 1e-8        # 防止除零

def compute_attention_importance(model, dataloader, device):
    """计算 Transformer Q, K, V 的注意力重要性"""
    num_layers = len(model.model.layers)
    module_importance = {
        "q_proj": np.zeros(num_layers),
        "k_proj": np.zeros(num_layers),
        "v_proj": np.zeros(num_layers)
    }

    with torch.no_grad():
        for batch in dataloader:
            # 将字典中的每个 tensor 转移到 device 上
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**inputs, output_attentions=True)
            attentions = outputs.attentions  # shape: (num_layers, batch_size, num_heads, seq_len, seq_len)

            for layer_idx, layer_attention in enumerate(attentions):
                # 对当前层的注意力分数取平均（batch、heads、seq_len）
                layer_attn_scores = layer_attention.mean(dim=(0, 2, 3)).cpu().numpy()
                module_importance["q_proj"][layer_idx] += layer_attn_scores.mean()
                module_importance["k_proj"][layer_idx] += layer_attn_scores.mean()
                module_importance["v_proj"][layer_idx] += layer_attn_scores.mean()

    for key in module_importance.keys():
        module_importance[key] /= len(dataloader)
    return module_importance

def compute_module_gradient_sensitivity(model):
    """计算 Transformer O, Gate, Up, Down 的梯度敏感性"""
    sensitivity_scores = {
        "o_proj": [],
        "gate_proj": [],
        "up_proj": [],
        "down_proj": []
    }

    for layer in model.model.layers:
        module_scores = {}
        for name, module in layer.named_modules():
            if any(key in name for key in sensitivity_scores.keys()):
                if module.weight.grad is not None:
                    score = torch.sum(torch.abs(module.weight * module.weight.grad)).item()
                    module_scores[name] = score

        total = sum(module_scores.values()) + EP
        for key in module_scores:
            module_scores[key] /= total

        for key in sensitivity_scores.keys():
            sensitivity_scores[key].append(module_scores.get(key, 0))
    return sensitivity_scores

def compute_importance_scores(model, dataloader, device):
    """计算最终的重要性分数，输出 1D 数组"""
    num_layers = len(model.model.layers)
    attn_scores = compute_attention_importance(model, dataloader, device)
    grad_scores = compute_module_gradient_sensitivity(model)

    importance_scores = []
    for i in range(num_layers):
        # 处理 Q, K, V 注意力重要性分数
        layer_attn = np.array([
            attn_scores["q_proj"][i],
            attn_scores["k_proj"][i],
            attn_scores["v_proj"][i]
        ])
        min_attn, max_attn = layer_attn.min(), layer_attn.max()
        mapped_attn = ((layer_attn - min_attn) / (max_attn - min_attn + EP)) * (S2 - S1) + S1
        mapped_attn *= ATTN_WEIGHT

        # 处理 O, Gate, Up, Down 的梯度敏感性分数
        layer_grad = np.array([
            grad_scores["o_proj"][i],
            grad_scores["gate_proj"][i],
            grad_scores["up_proj"][i],
            grad_scores["down_proj"][i]
        ])
        min_grad, max_grad = layer_grad.min(), layer_grad.max()
        mapped_grad = ((layer_grad - min_grad) / (max_grad - min_grad + EP)) * (S2 - S1) + S1
        mapped_grad *= MLP_WEIGHT

        # 拼接后每层得到 7 个分数（3 个注意力 + 4 个梯度）
        combined_scores = np.concatenate([mapped_attn, mapped_grad])
        importance_scores.extend(combined_scores)
    
    # 返回一个 1D 数组，长度为 num_layers * 7
    return np.array(importance_scores)

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

    # 加载 tokenizer（确保与模型一致）
    tokenizer = LlamaTokenizer.from_pretrained("HuggingFaceM4/llama-7b-tokenizer")
    
    # 生成一个测试样本
    text = "Hello, this is a test input for importance computation."
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    dataloader = [inputs]  # 这里假设 dataloader 只有一个 batch
    
    importance_scores = compute_importance_scores(model, dataloader, device)
    np.save("importance_scores.npy", importance_scores)
    print("Importance scores saved to importance_scores.npy")
