import numpy as np
metric = np.load("/root/pruning/AlphaPruning/data/llama-3-8b/alpha_peak.npy")

res = [[]*7]
for i in range(32):
    for j in range(7):
        res[j].append(metric[i*7 + j])

res = np.array(res)
for j in range(7):
    print("layer: ", j, " alphahill: ", np.mean(res[j]))

