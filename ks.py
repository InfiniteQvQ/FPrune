import torch
import numpy as np
from transformers import AutoModelForCausalLM, LlamaTokenizer

# 超参数
ATTN_WEIGHT = 1.5   # Q, K, V 的权重
MLP_WEIGHT = 1.0    # O, Gate, Up, Down 的权重
S1 = 0.9            # 线性映射下限
S2 = 1.0            # 线性映射上限
EP = 1e-8           # 防止除零

# 全局字典，用于存储每层 Q、K、V 的 hook 输出（这里统计输出张量的平均绝对值）
qkv_outputs = {"q": {}, "k": {}, "v": {}}

def get_hook(layer_index, name):
    """
    返回一个 hook 函数，用于记录指定层、指定模块（q, k, v）的输出统计值。
    这里用输出张量的平均绝对值作为重要性指标。
    """
    def hook(module, input, output):
        # 计算平均绝对值
        value = output.detach().abs().mean().item()
        if layer_index not in qkv_outputs[name]:
            qkv_outputs[name][layer_index] = []
        qkv_outputs[name][layer_index].append(value)
    return hook

def register_hooks(model):
    """
    在每一层的 self_attn 模块上注册 hook，捕获 q_proj, k_proj, v_proj 的输出。
    注意：此处假设每层的结构为 model.model.layers[i].self_attn.{q_proj, k_proj, v_proj}，
    如果你的模型模块名称不同，请相应调整。
    """
    for i, layer in enumerate(model.model.layers):
        try:
            layer.self_attn.q_proj.register_forward_hook(get_hook(i, "q"))
            layer.self_attn.k_proj.register_forward_hook(get_hook(i, "k"))
            layer.self_attn.v_proj.register_forward_hook(get_hook(i, "v"))
        except AttributeError:
            print(f"Layer {i} 未找到预期的 q,k,v 模块。")

def compute_qkv_importance():
    """
    对每一层的 hook 输出计算平均值，作为 Q, K, V 的原始重要性指标。
    """
    importance = {"q": [], "k": [], "v": []}
    for key in ["q", "k", "v"]:
        for i in sorted(qkv_outputs[key].keys()):
            # 平均所有 hook 记录值
            val = np.mean(qkv_outputs[key][i])
            importance[key].append(val)
    return importance

def compute_module_gradient_sensitivity(model, inputs, device):
    """
    计算每一层中 O, Gate, Up, Down 模块的梯度敏感性。
    为了获得非零梯度，这里先进行一次前向传播并利用 dummy loss (这里取 logits 的和)
    调用 backward() 以获得真实的梯度。
    """
    model.zero_grad()
    outputs = model(**inputs)
    # 定义一个 dummy loss（例如 logits 全部求和）
    loss = outputs.logits.sum()
    loss.backward()

    sensitivity_scores = {"o_proj": [], "gate_proj": [], "up_proj": [], "down_proj": []}
    for i, layer in enumerate(model.model.layers):
        module_scores = {}
        # 遍历每一层中的所有子模块，找到名称中包含目标关键字的模块
        for name, module in layer.named_modules():
            for key in sensitivity_scores.keys():
                if key in name and hasattr(module, 'weight') and module.weight.grad is not None:
                    # 计算权重与梯度乘积的绝对值之和作为敏感性指标
                    score = torch.sum(torch.abs(module.weight * module.weight.grad)).item()
                    module_scores[key] = score
        total = sum(module_scores.values()) + EP
        # 对每个模块计算归一化得分
        for key in sensitivity_scores.keys():
            val = module_scores.get(key, 0) / total
            sensitivity_scores[key].append(val)
    return sensitivity_scores

def compute_importance_scores(model, dataloader, device):
    """
    综合两部分：  
    1. 利用 hook 捕获各层 Q, K, V 的输出特征并计算重要性  
    2. 利用 forward-backward 得到 MLP 部分 (O, Gate, Up, Down) 的梯度敏感性  
    然后分别归一化（使用 S1, S2 做线性映射）并加权，最终每层得到 7 个分数，
    拼接成一个 1D 数组返回。
    """
    # 1. 注册 hook（注意：只注册一次即可）
    register_hooks(model)
    
    # 2. 前向传播，收集所有层的 Q, K, V 输出信息
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            inputs = {k: v.to(device) for k, v in batch.items()}
            model(**inputs)
    
    # 3. 计算 Q, K, V 的原始重要性
    qkv_importance = compute_qkv_importance()
    # 对每个模块独立归一化，并乘以 ATTN_WEIGHT
    mapped_qkv = {}
    for key in ["q", "k", "v"]:
        arr = np.array(qkv_importance[key])
        min_val, max_val = arr.min(), arr.max()
        mapped = ((arr - min_val) / (max_val - min_val + EP)) * (S2 - S1) + S1
        mapped_qkv[key] = mapped * ATTN_WEIGHT

    # 4. 对梯度敏感性部分进行计算
    # 这里取 dataloader 的第一个 batch 进行梯度计算（确保此 batch足以获得代表性梯度）
    batch = dataloader[0]
    inputs = {k: v.to(device) for k, v in batch.items()}
    grad_scores = compute_module_gradient_sensitivity(model, inputs, device)
    mapped_grad = {}
    for key in grad_scores.keys():
        arr = np.array(grad_scores[key])
        min_val, max_val = arr.min(), arr.max()
        mapped = ((arr - min_val) / (max_val - min_val + EP)) * (S2 - S1) + S1
        mapped_grad[key] = mapped * MLP_WEIGHT

    # 5. 拼接每层的 7 个分数（顺序：q, k, v, o_proj, gate_proj, up_proj, down_proj）
    num_layers = len(model.model.layers)
    importance_scores = []
    for i in range(num_layers):
        layer_scores = []
        # Q, K, V 分数
        layer_scores.extend([mapped_qkv["q"][i], mapped_qkv["k"][i], mapped_qkv["v"][i]])
        # 梯度敏感性分数：O, Gate, Up, Down
        layer_scores.extend([mapped_grad["o_proj"][i],
                             mapped_grad["gate_proj"][i],
                             mapped_grad["up_proj"][i],
                             mapped_grad["down_proj"][i]])
        importance_scores.extend(layer_scores)
    return np.array(importance_scores)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 加载 LLaMA 7B 模型（注意调整 cache_dir 和模型名称）
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
    # 此处构造一个简易的 dataloader，仅包含一个 batch
    dataloader = [inputs]
    
    # 计算重要性分数
    importance_scores = compute_importance_scores(model, dataloader, device)
    np.save("importance_scores.npy", importance_scores)
    print(importance_scores)
    print("Importance scores saved to importance_scores.npy")
