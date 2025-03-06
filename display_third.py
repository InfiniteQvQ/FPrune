import json
import numpy as np
import pandas as pd

# 读取 JSON 数据
with open("third_order_esd_results.json", "r") as file:
    esd_data = json.load(file)

# 解析数据
layer_values = {}
for key, value in esd_data.items():
    if key.startswith("model.layers"):
        layer_index = int(key.split(".")[2])  # 获取层号
        if layer_index not in layer_values:
            layer_values[layer_index] = []
        layer_values[layer_index].append(value)

# 计算每层的平均重要性
layer_importance = {layer: np.mean(values) for layer, values in layer_values.items()}

# 转换为 DataFrame 并排序
df_layer_importance = pd.DataFrame(layer_importance.items(), columns=["Layer", "Average Importance"])
df_layer_importance = df_layer_importance.sort_values(by="Layer")
importance_scores = np.array(df_layer_importance["Average Importance"].values, dtype=np.float32)

# 归一化参数
s1, s2 = 0.1, 0.9  # 线性缩放范围，可调整
sparsity_ratio = 0.7  # 70% 剪枝比例

# 计算归一化剪枝比例
max_score = np.max(importance_scores)
min_score = np.min(importance_scores)
layerwise_pruning_ratios_esd = (((importance_scores - min_score) / (max_score - min_score)) * (s2 - s1) + s1)

# 假设所有层都可剪枝（prunables 全部为 1）
prunables_tensor = np.ones_like(importance_scores, dtype=np.float32)

# 计算缩放因子
scaler = np.sum(prunables_tensor) * sparsity_ratio / (np.sum(prunables_tensor * layerwise_pruning_ratios_esd))
layerwise_pruning_ratios_esd = layerwise_pruning_ratios_esd * scaler

# 转换为列表格式
layerwise_pruning_ratios_esd = layerwise_pruning_ratios_esd.tolist()

# 输出剪枝比例
print(layerwise_pruning_ratios_esd)