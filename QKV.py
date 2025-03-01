default_weight = 0.1428571428571429
baseline_value =  24.140140533447266
# 原始数值
raw_values = [23.4, 23.72, 23.82, 23.98, 24.46, 24.37, 24.21]
components = ["Q", "K", "V", "Output", "Gate", "Up", "Down"]

# 计算每个组件与基准值的差异
differences = [v - baseline_value for v in raw_values]

# 计算平均差异
mean_diff = sum(differences) / len(differences)

# 设置缩放因子 s，保证调整后权重不会相差太大，最好使最高不超过0.15
s = 0.003

# 计算每个组件的权重
weights = [default_weight - s * (d - mean_diff) for d in differences]

# 输出结果
for comp, w in zip(components, weights):
    print(f"{comp}: {w:.6f}")

print("Sum:", sum(weights))
