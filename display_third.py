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

# 输出 DataFrame
print(df_layer_importance)
