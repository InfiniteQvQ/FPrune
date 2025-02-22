


import numpy as np
importance = np.array(a = [5.4375, 2.1562, 1.5625, 1.5625, 0.8836, 0.8836, 0.8836, 0.8836, 0.8836, 0.3727,0.3727,0.3727,0.3727,0.3727,0.3727,
        0.2061,0.2061,0.2061,0.1548,0.1548,0.1548,0.1548,0.1237,0.1237,0.1237,0.1138,0.1011,0.1011,0.1016,0.1069,0.0557,0.0415])
print(len(importance))
I_min = np.min(importance)
I_max = np.max(importance)
print("I_min =", I_min, " I_max =", I_max)

# 归一化
norm_importance = (importance - I_min) / (I_max - I_min)
# 反转：重要性高对应剪枝比例低
pre_ratio = 1 - norm_importance

# 计算预期的平均值
avg_pre_ratio = np.mean(pre_ratio)
print("Preliminary ratios:", pre_ratio)
print("Average of preliminary ratios:", avg_pre_ratio)

# 调整因子：使得平均值为0.5
target_avg = 0.5
scale_factor = target_avg / avg_pre_ratio
final_ratios = pre_ratio * scale_factor

# 若有超过1的，则 clip
final_ratios = np.clip(final_ratios, 0.0, 1.0)

print("Final pruning ratios:", final_ratios)
print("Mean final ratio:", np.mean(final_ratios))