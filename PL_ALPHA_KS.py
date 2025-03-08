import torch
import numpy as np
from scipy.stats import gaussian_kde, wasserstein_distance
from sklearn.feature_selection import mutual_info_regression
from transformers import LlamaModel, AutoTokenizer

def compute_pdf(activations, num_points=100):
    kde = gaussian_kde(activations)
    x_range = np.linspace(min(activations), max(activations), num_points)
    pdf_values = kde(x_range)
    return x_range, pdf_values

def compute_mutual_information(X, Y):
    X = X.reshape(X.shape[0], -1)
    Y = Y.reshape(Y.shape[0], -1)
    mi_values = []
    for i in range(X.shape[1]):
        mi = mutual_info_regression(X, Y[:, i])
        mi_values.append(np.mean(mi))
    return np.mean(mi_values)

def compute_spectral_entropy(H):
    H = H - np.mean(H, axis=0)
    C = np.cov(H, rowvar=False)
    eigvals = np.linalg.eigvalsh(C)
    eigvals = eigvals[eigvals > 1e-6]
    return -np.sum(eigvals * np.log(eigvals))

def compute_wasserstein_distance(H1, H2):
    H1 = H1.flatten()
    H2 = H2.flatten()
    return wasserstein_distance(H1, H2)

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

text = ["LLaMA 7B Analysis"]
inputs = tokenizer(text, return_tensors="pt")
inputs.pop("token_type_ids", None)
inputs = {key: val.to(device) for key, val in inputs.items()}

with torch.no_grad():
    outputs = model(**inputs)

hidden_states = outputs.hidden_states  # (num_layers, batch, seq_len, hidden_dim)

importance_scores = {}

for layer_idx in range(model.config.num_hidden_layers - 1):
    H = hidden_states[layer_idx][0].cpu().numpy()
    H_next = hidden_states[layer_idx + 1][0].cpu().numpy()

    # 计算 PDF/CCDF
    activations = H.flatten()
    x_range, pdf_values = compute_pdf(activations)

    # 计算 互信息
    mi = compute_mutual_information(H, H_next)

    # 计算特征值熵
    entropy = compute_spectral_entropy(H)

    # 计算 Wasserstein 距离
    wd = compute_wasserstein_distance(H, H_next)

    importance_scores[layer_idx] = {
        "Layer Importance": np.trapz(np.abs(x_range) * pdf_values, x_range),
        "Mutual Information": mi,
        "Spectral Entropy": entropy,
        "Wasserstein Distance": wd
    }

# 输出层的重要性
for layer, metrics in importance_scores.items():
    print(f"Layer {layer}:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
