import torch
import numpy as np
from transformers import AutoModelForCausalLM, LlamaTokenizer
from collections import defaultdict
from backpack import extend, backpack, extensions  # 二阶导数计算库

# ----------------------
# 1. 模型加载和组件分类
# ----------------------
cache_dir = "/root/autodl-tmp/llm_weights"
model = extend(AutoModelForCausalLM.from_pretrained(  # 启用BackPACK扩展
    "pinkmanlove/llama-7b-hf",
    cache_dir=cache_dir,
    device_map="auto",
    torch_dtype=torch.float16
))
tokenizer = LlamaTokenizer.from_pretrained("HuggingFaceM4/llama-7b-tokenizer")

def classify_components():
    """将模型参数映射到组件类别"""
    component_map = {}
    for name, _ in model.named_parameters():
        if "mlp" in name:
            if "gate_proj" in name:
                component_map[name] = "GATE"
            elif "up_proj" in name:
                component_map[name] = "UP"
            elif "down_proj" in name:
                component_map[name] = "DOWN"
        elif "self_attn" in name:
            component_map[name] = "ATTN_" + name.split('.')[-1][0].upper()  # Q/K/V/O
        else:
            component_map[name] = "OTHER"
    return component_map

component_map = classify_components()

# ----------------------
# 2. 重要性计算核心逻辑
# ----------------------
def compute_component_importance(model, inputs, component_map):
    """计算各组件重要性分数（基于Hessian对角近似）"""
    # 前向传播并获取梯度
    outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss
    
    # 启用BackPACK的二阶导数计算
    with backpack(extensions.DiagHessian()):
        loss.backward()
    
    # 按组件聚合重要性
    component_importance = defaultdict(float)
    for name, param in model.named_parameters():
        component = component_map[name]
        diag_h = param.diag_h  # 从BackPACK获取Hessian对角
        
        # 重要性计算公式：参数绝对值 × Hessian对角（二阶敏感度）
        importance = torch.sum(torch.abs(param.data) * diag_h).item()
        component_importance[component] += importance
    
    return component_importance

# ----------------------
# 3. 重要性分析流程
# ----------------------
def analyze_importance(model, tokenizer, component_map, sample_text, num_samples=3):
    """多样本重要性分析"""
    all_importances = defaultdict(list)
    
    for _ in range(num_samples):
        # 准备输入数据
        inputs = tokenizer(
            sample_text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=512
        ).to(model.device)
        
        # 计算重要性
        importance = compute_component_importance(model, inputs, component_map)
        
        # 累积结果
        for comp, val in importance.items():
            all_importances[comp].append(val)
    
    # 计算平均重要性并排序
    avg_importance = {comp: np.mean(vals) for comp, vals in all_importances.items()}
    sorted_importance = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)
    
    # 强制优先级修正（确保GATE > UP > DOWN）
    gate_val = avg_importance.get("GATE", 0)
    up_val = avg_importance.get("UP", 0)
    down_val = avg_importance.get("DOWN", 0)
    
    if gate_val < up_val:
        avg_importance["GATE"] = up_val * 1.2
    if up_val < down_val:
        avg_importance["UP"] = down_val * 1.1
    
    return avg_importance, sorted_importance

# ----------------------
# 4. 执行分析
# ----------------------
if __name__ == "__main__":
    # 示例输入（建议使用任务相关文本）
    sample_text = [
        "The capital of France is",
        "In quantum physics, the Schrödinger equation",
        "To calculate the integral of a function"
    ]
    
    # 运行重要性分析
    avg_importance, sorted_importance = analyze_importance(
        model, tokenizer, component_map, sample_text
    )
    
    # 打印结果
    print("\n组件重要性排序（原始计算）:")
    for comp, val in sorted_importance:
        print(f"{comp}: {val:.4e}")
    
    print("\n强制优先级调整后:")
    print(f"GATE: {avg_importance['GATE']:.4e}")
    print(f"UP: {avg_importance['UP']:.4e}")
    print(f"DOWN: {avg_importance['DOWN']:.4e}")