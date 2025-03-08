import numpy as np
import torch

kta_values =np.array([0.2533276000644673, 0.11248851424223413, 0.1275779949147944, 0.18277748396683965, 0.19408151959323844, 0.20919489505902292, 0.19963403038440447, 0.19452854306288056, 0.18895186755609555, 0.18268661576109096, 0.17895226220650753, 0.17983263055061602, 0.17536111562664072, 0.17139906934355564, 0.17235328301230857, 0.17742236081760354, 0.18567416463657477, 0.1754136964289023, 0.17273544834850554, 0.17052378841138627, 0.16653828295235967, 0.1592660940110598, 0.15082438282482108, 0.14620079590643567, 0.13631787690926828, 0.13219782324093138, 0.13096260028866819, 0.13084393187996252, 0.12843874613386633, 0.1306372311607941, 0.13066350964596862, 0.1402385089861504])
# Normalize the KTA values

kta_values = np.array([
    0.2533276000644673, 0.11248851424223413, 0.1275779949147944, 0.18277748396683965, 
    0.19408151959323844, 0.20919489505902292, 0.19963403038440447, 0.19452854306288056, 
    0.18895186755609555, 0.18268661576109096, 0.17895226220650753, 0.17983263055061602, 
    0.17536111562664072, 0.17139906934355564, 0.17235328301230857, 0.17742236081760354, 
    0.18567416463657477, 0.1754136964289023, 0.17273544834850554, 0.17052378841138627, 
    0.16653828295235967, 0.1592660940110598, 0.15082438282482108, 0.14620079590643567, 
    0.13631787690926828, 0.13219782324093138, 0.13096260028866819, 0.13084393187996252, 
    0.12843874613386633, 0.1306372311607941, 0.13066350964596862, 0.1402385089861504
])

# 归一化计算
normalized_kta = kta_values / np.sum(kta_values)

# 打印结果
for i, val in enumerate(normalized_kta):
    print(f"Layer {i}: {val:.4f}")

# 归一化 KTA 结果：
normalized_kta_list = normalized_kta.tolist()
print("Normalized KTA Importance:", normalized_kta_list)
kta_values = np.array(normalized_kta_list)
normalized_kta = (kta_values - np.min(kta_values)) / (np.max(kta_values) - np.min(kta_values))
normalized_kta = 1-normalized_kta
# Target sparsity range
s1, s2 = 0.8, 1.2  # Keeping original scale as in the given method

# Convert to torch tensor
scores = torch.tensor(normalized_kta, dtype=torch.float32)
prunables = torch.ones_like(scores)  # Assuming all layers are prunable

# Linear mapping
max_score = torch.max(scores)
min_score = torch.min(scores)
layerwise_pruning_ratios = (((scores - min_score) / (max_score - min_score)) * (s2 - s1) + s1)

# Adjusting sparsity ratio to target 70%
target_sparsity_ratio = 0.7  # 70%
scaler = torch.sum(prunables) * target_sparsity_ratio / (torch.sum(prunables * layerwise_pruning_ratios))
layerwise_pruning_ratios = layerwise_pruning_ratios * scaler

# Convert to list for output
layerwise_pruning_ratios = layerwise_pruning_ratios.cpu().numpy().tolist()

# Display the adjusted pruning ratios
print(layerwise_pruning_ratios)