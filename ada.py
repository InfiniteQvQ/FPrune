import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoConfig, AutoModelForCausalLM
from accelerate import Accelerator
import numpy as np
import matplotlib.pyplot as plt
import json

# 初始化加速器（自动处理多GPU分布）
accelerator = Accelerator()
device = accelerator.device

# 1. 分布式加载模型
def load_model_on_multiple_gpus():
    cache_dir = "/root/autodl-tmp/llm_weights"
    model = AutoModelForCausalLM.from_pretrained(
        "pinkmanlove/llama-7b-hf",
        cache_dir=cache_dir,
        device_map="auto",  # 让 Hugging Face 自动分配多个 GPU
        torch_dtype=torch.float16
    )


    return model

tokenizer_name = "HuggingFaceM4/llama-7b-tokenizer"
tokenizer = LlamaTokenizer.from_pretrained(tokenizer_name)
model = load_model_on_multiple_gpus()
model.eval()

# 2. 分布式数据准备
class AnalysisDataset(torch.utils.data.Dataset):
    def __init__(self, texts, tokenizer):
        self.encodings = tokenizer(texts, padding=True, return_tensors="pt")
        
    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}
    
    def __len__(self):
        return len(self.encodings.input_ids)

texts = ["The capital of France is", "Machine learning is"]  # 示例输入
dataset = AnalysisDataset(texts, tokenizer)
dataloader = DataLoader(dataset, batch_size=2)
model, dataloader = accelerator.prepare(model, dataloader)

# 3. 分析核心函数
def analyze_contributions(model, dataloader):
    gradnorms = {n: 0.0 for n, _ in model.named_parameters()}
    alphahills = {n: 0.0 for n, _ in model.named_parameters()}
    sample_count = 0
    
    with torch.no_grad():
        for batch in dataloader:
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            
            # 梯度计算
            accelerator.backward(loss)
            
            # 一阶项统计
            for name, param in model.named_parameters():
                if param.grad is not None:
                    gradnorms[name] += accelerator.gather(param.grad.norm()).mean().item()
            
            # 二阶项近似（分块计算）
            for name, param in model.named_parameters():
                if param.grad is not None:
                    hessian_approx = []
                    grad = param.grad.detach().flatten()
                    
                    # 分块处理防止OOM
                    for i in range(0, grad.shape[0], 512):
                        grad_slice = grad[i:i+512]
                        if grad_slice.shape[0] == 0:
                            continue
                            
                        hessian_slice = torch.autograd.grad(
                            grad_slice.sum(), 
                            param, 
                            retain_graph=True,
                            allow_unused=True
                        )[0]
                        
                        if hessian_slice is not None:
                            hessian_approx.append(hessian_slice.abs().mean().item())
                    
                    if hessian_approx:
                        alphahills[name] += np.mean(hessian_approx)
            
            sample_count += 1
            model.zero_grad()
    
    # 平均计算结果
    return {k: v/sample_count for k, v in gradnorms.items()}, {k: v/sample_count for k, v in alphahills.items()}

# 4. 运行分析
gradnorms, alphahills = analyze_contributions(model, dataloader)

# 5. 重要性综合评估
def compute_combined_importance(gradnorms, alphahills):
    importance = {}
    max_grad = max(gradnorms.values())
    max_alpha = max(alphahills.values())
    
    for name in gradnorms.keys():
        # 标准化指标
        norm_grad = gradnorms[name] / max_grad
        norm_alpha = alphahills[name] / max_alpha
        
        # 综合重要性公式（可调整权重）
        importance[name] = 0.7 * norm_grad + 0.3 * norm_alpha
    
    return importance

combined_importance = compute_combined_importance(gradnorms, alphahills)

# 6. 结果可视化
def plot_dual_analysis(gradnorms, alphahills, top_n=15):
    # 按综合重要性排序
    sorted_names = sorted(combined_importance, key=lambda x: combined_importance[x], reverse=True)[:top_n]
    
    # 准备数据
    grad_values = [gradnorms[name] for name in sorted_names]
    alpha_values = [alphahills[name] for name in sorted_names]
    
    # 绘制双轴图
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()
    
    ax1.bar(range(len(sorted_names)), grad_values, color='b', alpha=0.6, label='GradNorm')
    ax2.plot(alpha_values, color='r', marker='o', label='AlphaHill')
    
    ax1.set_xlabel('Layers')
    ax1.set_ylabel('GradNorm', color='b')
    ax2.set_ylabel('AlphaHill', color='r')
    plt.title('Layer Importance Analysis')
    plt.xticks(range(len(sorted_names)), sorted_names, rotation=90)
    fig.tight_layout()
    plt.show()

plot_dual_analysis(gradnorms, alphahills)

# 7. 结果保存
with open("dual_gpu_analysis.json", "w") as f:
    json.dump({
        "gradnorms": gradnorms,
        "alphahills": alphahills,
        "combined_importance": combined_importance
    }, f, indent=2)