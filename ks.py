import torch
import numpy as np
from transformers import AutoModelForCausalLM, LlamaTokenizer

# 超参数：用于线性映射和防止除零
S1 = 0.9
S2 = 1.0
EP = 1e-8

def compute_finite_difference_sensitivity(model, inputs, module_key, epsilon=1e-5):
    """
    对于模型中名称包含 module_key 的参数，利用有限差分法计算敏感性指标。
    具体做法是：对该模块的参数加上微小扰动，然后计算模型输出（例如 logits）的变化幅度。
    
    返回一个字典，键为参数名称，值为该参数扰动后模型输出变化的归一化指标。
    """
    # 保存模型原始输出（这里选择 logits 作为输出）
    outputs_original = model(**inputs).logits.detach()
    # 记录所有敏感性指标
    sensitivities = {}
    
    # 遍历模型所有参数
    for name, param in model.named_parameters():
        if module_key in name:
            # 保存原始参数
            original = param.data.clone()
            # 为了确保数值稳定，先将参数置为 float32 计算敏感性
            param.data = param.data.float()
            # 对参数加上 epsilon 扰动（这里采用正向扰动，可以扩展为正负扰动的平均）
            param.data.add_(epsilon)
            # 计算扰动后的输出
            outputs_perturbed = model(**inputs).logits.detach()
            # 计算输出变化（例如采用 Frobenius 范数衡量所有输出变化）
            delta = torch.norm(outputs_perturbed - outputs_original)
            # 将变化值归一化（例如除以 epsilon，使其近似于数值梯度的范数）
            sensitivity = delta / epsilon
            sensitivities[name] = sensitivity.item()
            # 恢复原始参数
            param.data.copy_(original)
    
    return sensitivities

def compute_all_module_sensitivities(model, inputs, module_keys):
    """
    对给定的多个 module_key（例如 ["o_proj", "gate_proj", "up_proj", "down_proj"]），
    计算各模块基于有限差分的敏感性指标，并对每层进行归一化。
    
    返回一个字典：键为模块名称，值为每层的敏感性数组（每层一个数值）。
    """
    # 初始化字典，假设模型有 L 层
    num_layers = len(model.model.layers)
    sensitivity_scores = {key: [0] * num_layers for key in module_keys}
    
    # 对 dataloader 的一个 batch 计算（确保 inputs 在 device 上）
    for key in module_keys:
        # 对每个模块 key，在各个层中寻找参数（假设参数名称中包含 "model.layers.{i}"）
        for i in range(num_layers):
            layer_key = f"model.layers.{i}"
            # 筛选出属于当前层且名称中包含 module key 的参数
            layer_params = {name: param for name, param in model.named_parameters()
                            if layer_key in name and key in name}
            if len(layer_params) == 0:
                # 如果当前层没有找到对应模块，记录为 0
                sensitivity_scores[key][i] = 0.0
            else:
                # 对当前层的每个参数分别计算敏感性，然后取平均
                sens_list = []
                for name, param in layer_params.items():
                    sens = compute_finite_difference_sensitivity(model, inputs, name, epsilon=1e-5)
                    # 这里 sens 是一个字典，但由于只针对当前参数，取该参数对应的数值
                    if name in sens:
                        sens_list.append(sens[name])
                # 如果 sens_list 非空，则取平均，否则为 0
                if sens_list:
                    sensitivity_scores[key][i] = np.mean(sens_list)
                else:
                    sensitivity_scores[key][i] = 0.0
    return sensitivity_scores

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 加载模型和 tokenizer（请根据实际情况修改模型名称和路径）
    cache_dir = "/root/autodl-tmp/llm_weights"
    model = AutoModelForCausalLM.from_pretrained(
        "pinkmanlove/llama-7b-hf",
        cache_dir=cache_dir,
        device_map="auto",
        torch_dtype=torch.float16
    )
    model.to(device)
    
    tokenizer = LlamaTokenizer.from_pretrained("HuggingFaceM4/llama-7b-tokenizer")
    
    # 构造输入
    text = "Hello, this is a test input for sensitivity computation."
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # 我们希望计算以下模块的敏感性
    module_keys = ["o_proj", "gate_proj", "up_proj", "down_proj"]
    
    # 计算有限差分敏感性
    sensitivities = compute_all_module_sensitivities(model, inputs, module_keys)
    
    # 为方便观察，可以对每层的各模块敏感性归一化映射到 [S1, S2]
    mapped_sensitivities = {}
    for key, scores in sensitivities.items():
        arr = np.array(scores)
        min_val, max_val = arr.min(), arr.max()
        mapped = ((arr - min_val) / (max_val - min_val + EP)) * (S2 - S1) + S1
        mapped_sensitivities[key] = mapped
    
    # 拼接每层的敏感性分数（假设模型有 L 层，每层4个指标）
    num_layers = len(model.model.layers)
    final_scores = []
    for i in range(num_layers):
        layer_scores = []
        for key in module_keys:
            layer_scores.append(mapped_sensitivities[key][i])
        final_scores.extend(layer_scores)
    
    final_scores = np.array(final_scores)
    np.save("importance_scores.npy", final_scores)
    print("Importance scores (finite-difference based) saved to importance_scores.npy")
    print(final_scores)
