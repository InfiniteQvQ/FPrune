import numpy as np
importance = np.array([4.0938, 3.2188, 2.5, 1.9590,1.9590,1.9590,1.9590, 1.6719, 1.6719, 1.6719, 1.6719, 1.6719, 1.6719, 1.4766, 1.3438, 
                       1.2461, 1.2461,1.0820,1.0820, 0.8174,0.8174,0.8174,0.8174, 0.5762, 0.5762, 0.4531, 0.1699, 0.0952])
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