import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from transformers import LlamaModel, AutoTokenizer

def compute_pdf(activations, num_points=100):
    """计算 PDF（概率密度函数）"""
    kde = gaussian_kde(activations)
    x_range = np.linspace(min(activations), max(activations), num_points)
    pdf_values = kde(x_range)
    return x_range, pdf_values

def compute_ccdf(activations):
    """计算 CCDF（补充累积分布函数）"""
    sorted_activations = np.sort(activations)
    ccdf_values = 1 - np.arange(1, len(sorted_activations) + 1) / len(sorted_activations)
    return sorted_activations, ccdf_values

def compute_layer_importance(x_range, pdf_values):
    """计算层的重要性 (Layer Importance)"""
    return np.trapz(np.abs(x_range) * pdf_values, x_range)  # 数值积分计算重要性

# **加载 LLaMA 7B 模型**
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "pinkmanlove/llama-7b-hf"

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceM4/llama-7b-tokenizer")
model = LlamaModel.from_pretrained(
    model_name,
    cache_dir="/root/autodl-tmp/llm_weights",
    output_hidden_states=True,
    torch_dtype=torch.float16,
    device_map="auto"
)

# **处理输入**
text = ["LLaMA 7B Activation Distribution Analysis."]
inputs = tokenizer(text, return_tensors="pt")
inputs.pop("token_type_ids", None)
inputs = {key: val.to(device) for key, val in inputs.items()}

# **获取隐藏状态**
with torch.no_grad():
    outputs = model(**inputs)

hidden_states = outputs.hidden_states  # (num_layers, batch, seq_len, hidden_dim)

layer_importance_scores = {}  # 存储层重要性得分
layer_pdf_data = {}  # 存储 PDF 数据
layer_ccdf_data = {}  # 存储 CCDF 数据

# **遍历所有层**
for layer_idx in range(model.config.num_hidden_layers):
    activations = hidden_states[layer_idx][0].cpu().numpy().flatten()

    # **计算 PDF**
    x_range, pdf_values = compute_pdf(activations)

    # **计算 CCDF**
    sorted_activations, ccdf_values = compute_ccdf(activations)

    # **计算层重要性**
    layer_importance = compute_layer_importance(x_range, pdf_values)
    layer_importance_scores[layer_idx] = layer_importance
    layer_pdf_data[layer_idx] = (x_range, pdf_values)
    layer_ccdf_data[layer_idx] = (sorted_activations, ccdf_values)

    print(f"Layer {layer_idx} Importance: {layer_importance:.4f}")

# **排序层重要性**
sorted_importance = sorted(layer_importance_scores.items(), key=lambda x: x[1], reverse=True)
print("\nFinal Layer Importance Ranking:")
for layer, score in sorted_importance:
    print(f"Layer {layer}: {score:.4f}")

