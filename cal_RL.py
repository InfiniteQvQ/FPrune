import os
import numpy as np
import torch
eps=1e-8
s1= 0.8
s2=1.2
file_path = "../AlphaPruning/data/llama-7b-hf/alpha_peak.npy"
metrics = np.load(file_path)
block_metrics = [np.mean(metrics[i:i+7])
                         for i in range(0, len(metrics), 7)]

metrics = [val for val in block_metrics for _ in range(7)]

scores = torch.tensor(metrics, dtype=torch.float32)

max_score = torch.max(scores)
min_score = torch.min(scores)

layerwise_pruning_ratios_esd = (((scores - min_score) / (max_score - min_score + eps)) * (s2 - s1) + s1)
    # 校正以满足整体稀疏率要求

layerwise_pruning_ratios_esd = layerwise_pruning_ratios_esd * 0.710043
layerwise_pruning_ratios_esd = layerwise_pruning_ratios_esd.cpu().numpy().tolist()
print("ESD-based ratios:", layerwise_pruning_ratios_esd)

layerwise_pruning_ratios_esd = [0.5293006074418442, 0.5293006074418442, 0.5293006074418442, 0.5293006074418442, 0.5293006074418442, 0.5293006074418442, 0.5293006074418442, 0.6358363452242717, 0.6358363452242717, 0.6358363452242717, 0.6358363452242717, 0.6358363452242717, 0.6358363452242717, 0.6358363452242717, 0.7748192144399018, 0.7748192144399018, 0.7748192144399018, 0.7748192144399018, 0.7748192144399018, 0.7748192144399018, 0.7748192144399018, 0.7234297956976186, 0.7234297956976186, 0.7234297956976186, 0.7234297956976186, 0.7234297956976186, 0.7234297956976186, 0.7234297956976186, 0.705076419482818, 0.705076419482818, 0.705076419482818, 0.705076419482818, 0.705076419482818, 0.705076419482818, 0.705076419482818, 0.7139799622267773, 0.7139799622267773, 0.7139799622267773, 0.7139799622267773, 0.7139799622267773, 0.7139799622267773, 0.7139799622267773, 0.747167659368869, 0.747167659368869, 0.747167659368869, 0.747167659368869, 0.747167659368869, 0.747167659368869, 0.747167659368869, 0.5980377426491537, 0.5980377426491537, 0.5980377426491537, 0.5980377426491537, 0.5980377426491537, 0.5980377426491537, 0.5980377426491537, 0.5390139233026938, 0.5390139233026938, 0.5390139233026938, 0.5390139233026938, 0.5390139233026938, 0.5390139233026938, 0.5390139233026938, 0.5062957468510644, 0.5062957468510644, 0.5062957468510644, 0.5062957468510644, 0.5062957468510644, 0.5062957468510644, 0.5062957468510644, 0.5670330803422492, 0.5670330803422492, 0.5670330803422492, 0.5670330803422492, 0.5670330803422492, 0.5670330803422492, 0.5670330803422492, 0.5246790027265411, 0.5246790027265411, 0.5246790027265411, 0.5246790027265411, 0.5246790027265411, 0.5246790027265411, 0.5246790027265411, 0.5556295724845334, 0.5556295724845334, 0.5556295724845334, 0.5556295724845334, 0.5556295724845334, 0.5556295724845334, 0.5556295724845334, 0.5940288875974529, 0.5940288875974529, 0.5940288875974529, 0.5940288875974529, 0.5940288875974529, 0.5940288875974529, 0.5940288875974529, 0.5995001962463322, 0.5995001962463322, 0.5995001962463322, 0.5995001962463322, 0.5995001962463322, 0.5995001962463322, 0.5995001962463322, 0.672528180351988, 0.672528180351988, 0.672528180351988, 0.672528180351988, 0.672528180351988, 0.672528180351988, 0.672528180351988, 0.7913333400142103, 0.7913333400142103, 0.7913333400142103, 0.7913333400142103, 0.7913333400142103, 0.7913333400142103, 0.7913333400142103, 0.7112298065103565, 0.7112298065103565, 0.7112298065103565, 0.7112298065103565, 0.7112298065103565, 0.7112298065103565, 0.7112298065103565, 0.6983272207225116, 0.6983272207225116, 0.6983272207225116, 0.6983272207225116, 0.6983272207225116, 0.6983272207225116, 0.6983272207225116, 0.7847523906736767, 0.7847523906736767, 0.7847523906736767, 0.7847523906736767, 0.7847523906736767, 0.7847523906736767, 0.7847523906736767, 0.7486794849375509, 0.7486794849375509, 0.7486794849375509, 0.7486794849375509, 0.7486794849375509, 0.7486794849375509, 0.7486794849375509, 0.7650636317350729, 0.7650636317350729, 0.7650636317350729, 0.7650636317350729, 0.7650636317350729, 0.7650636317350729, 0.7650636317350729, 0.8405095219304548, 0.8405095219304548, 0.8405095219304548, 0.8405095219304548, 0.8405095219304548, 0.8405095219304548, 0.8405095219304548, 0.8392653829077255, 0.8392653829077255, 0.8392653829077255, 0.8392653829077255, 0.8392653829077255, 0.8392653829077255, 0.8392653829077255, 0.8146051974336859, 0.8146051974336859, 0.8146051974336859, 0.8146051974336859, 0.8146051974336859, 0.8146051974336859, 0.8146051974336859, 0.7603679049430712, 0.7603679049430712, 0.7603679049430712, 0.7603679049430712, 0.7603679049430712, 0.7603679049430712, 0.7603679049430712, 0.7766239792175328, 0.7766239792175328, 0.7766239792175328, 0.7766239792175328, 0.7766239792175328, 0.7766239792175328, 0.7766239792175328, 0.8103622440266844, 0.8103622440266844, 0.8103622440266844, 0.8103622440266844, 0.8103622440266844, 0.8103622440266844, 0.8103622440266844, 0.9402635298662624, 0.9402635298662624, 0.9402635298662624, 0.9402635298662624, 0.9402635298662624, 0.9402635298662624, 0.9402635298662624, 0.7872863009674198, 0.7872863009674198, 0.7872863009674198, 0.7872863009674198, 0.7872863009674198, 0.7872863009674198, 0.7872863009674198, 0.6663283535177681, 0.6663283535177681, 0.6663283535177681, 0.6663283535177681, 0.6663283535177681, 0.6663283535177681, 0.6663283535177681, 0.6786439656583021, 0.6786439656583021, 0.6786439656583021, 0.6786439656583021, 0.6786439656583021, 0.6786439656583021, 0.6786439656583021]

b  = []
sum = 0
for i in range(32):
    b.append(layerwise_pruning_ratios_esd[i*7])
    sum += b[-1]

print(b)
print("mean: ", sum/(32))
segments = {
        0: [0],
        1: [1],
        2: [2],
        3: [3],
        4: [4, 5, 6, 7, 8, 9, 10, 11],
        5: [12, 13, 14],
        6: [15, 16, 17],
        7: [18, 19, 20],
        8: [21, 22, 23],
        9: [24, 25],
        10: [26, 27],
        11: [28, 29],
        12: [30],
        13: [31]
}


res = []
cur_pointer = 0
for seg, l in segments.items():
    lens = len(l)
    cur = 0
    
    for i in range(lens):
        cur += layerwise_pruning_ratios_esd[cur_pointer * 7]
        cur_pointer += 1
    cur /= lens
    for i in range(lens):
        for j in range(7):
            res.append(cur)

b  = []
sum = 0
for i in range(32):
    b.append(res[i*7])
    sum += b[-1]

print(b)
print("mean: ", sum/(32))