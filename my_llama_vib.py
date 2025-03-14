import numpy as np
metric = np.load("/root/pruning/AlphaPruning/data/llama-3-8b/alpha_peak.npy")
res = [[] for _ in range(7)]

for i in range(32):  # 遍历 32 层
    for j in range(7):  # 遍历 Q, K, V, O, Gate, Up, Down
        res[j].append(metric[i * 7 + j])

# 🎯 转换为 NumPy 数组（确保结构正确）
res = np.array(res, dtype=object)

for j, name in enumerate(["Q", "K", "V", "O", "Gate", "Up", "Down"]):
    print(f"{name} 平均 Alpha-Hill: {np.mean(res[j]):.4f}")
