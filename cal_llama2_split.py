


import numpy as np
importance = np.array(a = [5.4062, 2.1719, 1.7109, 1.4219, 1.1484, 1.0469, 0.8672, 
                           0.7461, 0.6172, 0.5234, 0.4512, 0.377, 0.3379, 0.2969, 0.2520,
                           0.2236, 0.2041, 0.1904,0.1699, 0.1611,0.1582, 0.1299,0.127,
                           0.125,0.1191, 0.1138,0.104,0.0981,0.1016, 0.1069,0.0557,0.0415])
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