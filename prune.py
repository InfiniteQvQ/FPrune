import time
import heapq
import torch
import torch.nn as nn
from .sparsegpt import SparseGPT 
from .layerwrapper import WrappedGPT
from .data import get_loaders 
import numpy as np
from collections import defaultdict
from typing import List, Dict
import os
from scipy.ndimage import gaussian_filter1d
import torch.nn as nn
from tqdm import tqdm
from .utils import get_weights, get_modules


def find_layers(module, layers=[nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def check_sparsity(model):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 


    if "OPT" in model.__class__.__name__:
        layers = model.model.decoder.layers
    else:    
        layers = model.model.layers  
        
        
    count = 0 
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            count += (W==0).sum().item()
            total_params += W.numel()

            sub_count += (W==0).sum().item()
            sub_params += W.numel()

        print(f"layer {i} sparsity {float(sub_count)/sub_params:.6f}")

    model.config.use_cache = use_cache 
    return float(count)/total_params 

def prepare_calibration_input(model, dataloader, device):
    layers = model.model.layers

    # dev = model.hf_device_map["model.embed_tokens"]
    if "model.embed_tokens" in model.hf_device_map:
        device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((128, model.seqlen, model.config.hidden_size), dtype=dtype, device=device)
    inps.requires_grad = False
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass 

    layers[0] = layers[0].module
    torch.cuda.empty_cache()
    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    return inps, outs, attention_mask, position_ids

def prepare_calibration_input_opt(model, dataloader, device):
    layers = model.model.decoder.layers
    
    if "model.embed_tokens" in model.hf_device_map:
        device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((128, model.seqlen, model.config.hidden_size), dtype=dtype, device=device)
    inps.requires_grad = False
    cache = {'i': 0, 'attention_mask': None}
    
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            raise ValueError
        
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass 
        
    layers[0] = layers[0].module
    torch.cuda.empty_cache()
    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']

    return inps, outs, attention_mask, None


def return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before):
    thres_cumsum = sum_before * alpha 
    sort_mask = tmp_metric <= thres_cumsum.reshape((-1,1))
    thres = torch.gather(sort_res[0], dim=1, index=sort_mask.sum(dim=1, keepdims=True)-1)
    W_mask = (W_metric <= thres)
    cur_sparsity = (W_mask==True).sum() / W_mask.numel()
    return W_mask, cur_sparsity


def ww_sparsity(args, model, device=torch.device("cuda:0"), s1=0.8, s2=1.2, ratios=None, prune_n=0, prune_m=0):
    if "opt" in args.model:
        blocks = model.model.decoder.layers    
    else:
        blocks = model.model.layers
    
    
    layers = [find_layers(blocks)]
    prunables = []
    for layer in layers:
        for name in layer:
            prunables.append(layer[name].weight.numel())

    layer_num_in_block = int(len(prunables) / len(blocks))

    metrics = np.load(f"{args.ww_metric_cache}/{args.ww_metric}.npy")
    print("metrics ", metrics)
    if args.mapping_type == 'block_wise':
        block_metrics = [np.mean(metrics[i:i+layer_num_in_block]) for i in range(0, len(metrics), layer_num_in_block)]
        metrics = [i for i in block_metrics for j in range(layer_num_in_block)]
    
    print("metric values:", metrics)
            
    scores = torch.tensor(metrics)
    prunables = torch.tensor(prunables)

    # linear mapping
    max = torch.max(scores)
    min = torch.min(scores)
    
    layerwise_pruning_ratios = (((scores - min) / (max - min)) * (s2 - s1) + s1)
    scaler = torch.sum(prunables) * args.sparsity_ratio / (torch.sum(prunables * layerwise_pruning_ratios))  
    layerwise_pruning_ratios = layerwise_pruning_ratios * scaler
    layerwise_pruning_ratios = layerwise_pruning_ratios.cpu().numpy().tolist()
    print("ratio: ")
    print(layerwise_pruning_ratios)
    #layerwise_pruning_ratios = np.clip(layerwise_pruning_ratios, 0.0, 1.0)
    #print(layerwise_pruning_ratios, " new ratio")
    return layerwise_pruning_ratios

def ww_sparsity_llama_32_3b(args, model, device=torch.device("cuda:0"),
                         s1=0.8, s2=1.2, ratios=None, prune_n=0, prune_m=0,
                         weight_esd=0.85, eps=1e-8):
    if "opt" in args.model:
        blocks = model.model.decoder.layers    
    else:
        blocks = model.model.layers

    # 得到待剪枝层字典，假设 find_layers 返回的顺序与 transformer 层顺序一致，
    # 每个 transformer 层内有7个子层
    layers = [find_layers(blocks)]
    prunables = []
    for layer in layers:
        for name in layer:
            prunables.append(layer[name].weight.numel())
    layer_num_in_block = int(len(prunables) / len(blocks))
    
    # 加载ESD指标
    metrics = np.load(f"{args.ww_metric_cache}/{args.ww_metric}.npy")

    if args.mapping_type == 'block_wise':
        block_metrics = [np.mean(metrics[i:i+layer_num_in_block]) 
                         for i in range(0, len(metrics), layer_num_in_block)]
        metrics = [i for i in block_metrics for j in range(layer_num_in_block)]
    print("ESD metric values after block_wise processing:", metrics)

    scores = torch.tensor(metrics)
    prunables = torch.tensor(prunables)

    # linear mapping
    max = torch.max(scores)
    min = torch.min(scores)
    
    layerwise_pruning_ratios = (((scores - min) / (max - min)) * (s2 - s1) + s1)
    scaler = torch.sum(prunables) * args.sparsity_ratio / (torch.sum(prunables * layerwise_pruning_ratios))  
    layerwise_pruning_ratios = layerwise_pruning_ratios * scaler
    layerwise_pruning_ratios = layerwise_pruning_ratios.cpu().numpy().tolist()
    print("esd:  ", layerwise_pruning_ratios)
    segments = {
        0: [0],
        1: [1],
        2: [2],
        3: [3, 4, 5, 6],
        4: [7, 8, 9, 10, 11, 12],
        5: [13],
        6: [14],
        7: [15, 16],
        8: [17, 18],
        9: [19, 20, 21, 22],
        10: [23, 24],
        11: [25],
        12: [26],
        13: [27]
    }



    res = []
    cur_pointer = 0
    for seg, l in segments.items():
        lens = len(l)
        cur = 0
        
        for i in range(lens):
            cur += layerwise_pruning_ratios[cur_pointer * 7]
            cur_pointer += 1
        cur /= lens
        for i in range(lens):
            for j in range(7):
                res.append(cur)
    print(res)
    k = res
    res = []

    for i in range(28):
        #Q
        res.append(k[i*7] *0.142638 * 7 )
        #K
        res.append(k[i*7]  *0.141982 * 7 )
        #V
        res.append(k[i*7]  * 0.148107 * 7 )
        #OUT
        res.append(k[i*7]  * 0.142638 * 7  )
        #GATE1
        res.append(k[i*7]  *0.142795  * 7   )
        #UP
        res.append(k[i*7]  *0.142795 * 7  )
        #DOWN

        res.append(k[i*7]  * 0.142795 * 7  )
    print(res)
    return res

    
def ww_sparsity_llama_13b(args, model, device=torch.device("cuda:0"),
                         s1=0.8, s2=1.2, ratios=None, prune_n=0, prune_m=0,
                         weight_esd=0.85, eps=1e-8):
    # ---------------------- 基于ESD的剪枝比例计算 ----------------------
    if "opt" in args.model:
        blocks = model.model.decoder.layers    
    else:
        blocks = model.model.layers

    # 得到待剪枝层字典，假设 find_layers 返回的顺序与 transformer 层顺序一致，
    # 每个 transformer 层内有7个子层
    layers = [find_layers(blocks)]
    prunables = []
    for layer in layers:
        for name in layer:
            prunables.append(layer[name].weight.numel())
    layer_num_in_block = int(len(prunables) / len(blocks))
    
    # 加载ESD指标
    metrics = np.load(f"{args.ww_metric_cache}/{args.ww_metric}.npy")

    if args.mapping_type == 'block_wise':
        block_metrics = [np.mean(metrics[i:i+layer_num_in_block]) 
                         for i in range(0, len(metrics), layer_num_in_block)]
        metrics = [i for i in block_metrics for j in range(layer_num_in_block)]
    print("ESD metric values after block_wise processing:", metrics)

    scores = torch.tensor(metrics)
    prunables = torch.tensor(prunables)

    # linear mapping
    max = torch.max(scores)
    min = torch.min(scores)
    
    layerwise_pruning_ratios = (((scores - min) / (max - min)) * (s2 - s1) + s1)
    scaler = torch.sum(prunables) * args.sparsity_ratio / (torch.sum(prunables * layerwise_pruning_ratios))  
    layerwise_pruning_ratios = layerwise_pruning_ratios * scaler
    layerwise_pruning_ratios = layerwise_pruning_ratios.cpu().numpy().tolist()
    print("esd:  ", layerwise_pruning_ratios)
    segments = {
        0: [0],
        1: [1],
        2: [2, 3],
        3: [4, 5],
        4: [6, 7, 8, 9],
        5: [10, 11],
        6: [12, 13],
        7: [14],
        8: [15, 16, 17, 18],
        9: [19],
        10: [20, 21, 22, 23, 24, 25, 26],
        11: [27, 28, 29],
        12: [30, 31, 32],
        13: [33],
        14: [34],
        15: [35],
        16: [36],
        17: [37],
        18: [38],
        19: [39]
    }



    res = []
    cur_pointer = 0
    for seg, l in segments.items():
        lens = len(l)
        cur = 0
        
        for i in range(lens):
            cur += layerwise_pruning_ratios[cur_pointer * 7]
            cur_pointer += 1
        cur /= lens
        for i in range(lens):
            for j in range(7):
                res.append(cur)
    print(res)

    return res


def ww_sparsity_llama2_7b(args, model, device=torch.device("cuda:0"),
                         s1=0.8, s2=1.2, ratios=None, prune_n=0, prune_m=0,
                         weight_esd=0.5, eps=1e-8):
    if "opt" in args.model:
        blocks = model.model.decoder.layers    
    else:
        blocks = model.model.layers

    # 得到待剪枝层字典，假设 find_layers 返回的顺序与 transformer 层顺序一致，
    # 每个 transformer 层内有7个子层
    layers = [find_layers(blocks)]
    prunables = []
    for layer in layers:
        for name in layer:
            prunables.append(layer[name].weight.numel())
    layer_num_in_block = int(len(prunables) / len(blocks))
    
    # 加载ESD指标
    metrics = np.load(f"{args.ww_metric_cache}/{args.ww_metric}.npy")
    print("ESD raw metrics:", metrics)
    if args.mapping_type == 'block_wise':
        block_metrics = [np.mean(metrics[i:i+layer_num_in_block]) 
                         for i in range(0, len(metrics), layer_num_in_block)]
        metrics = [i for i in block_metrics for j in range(layer_num_in_block)]
    print("ESD metric values after block_wise processing:", metrics)
            
    scores = torch.tensor(metrics, dtype=torch.float32)
    prunables_tensor = torch.tensor(prunables, dtype=torch.float32)
    max_score = torch.max(scores)
    min_score = torch.min(scores)
    # 线性映射到 [s1, s2]
    layerwise_pruning_ratios_esd = (((scores - min_score) / (max_score - min_score)) * (s2 - s1) + s1)
    scaler = torch.sum(prunables_tensor) * args.sparsity_ratio / (torch.sum(prunables_tensor * layerwise_pruning_ratios_esd))
    layerwise_pruning_ratios_esd = layerwise_pruning_ratios_esd * scaler
    layerwise_pruning_ratios_esd = layerwise_pruning_ratios_esd.cpu().numpy().tolist()
    print("ESD-based ratios:", layerwise_pruning_ratios_esd)
   
    segments = {
        0: [0],
        1: [1],
        2: [2,3],
        3: [4,5,6,7,8],
        4: [9, 10, 11, 12, 13, 14],
        5: [15, 16, 17],
        6: [18, 19, 20, 21],
        7: [22, 23, 24],
        8: [25],
        9: [26, 27],
        10: [28],
        11: [29],
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
    print(res)

    return res
def ww_sparsity_llama_7b_split(args, model, device=torch.device("cuda:0"),
                                s1=0.8, s2=1.2, ratios=None, prune_n=0, prune_m=0,
                                weight_esd=0.98, eps=1e-8):
    """
    基于 ESD 数值计算 LLaMA 7B 各层（32 层，每层 7 个模块：Q, K, V, Out, Gate, Up, Down）的剪枝比例。
    计算流程：
      1. 加载 ESD 指标，并按照分块（segmentation）计算 block-level 平均 ESD；
      2. 计算线性映射得到每个模块的剪枝比例（总长度 224）；
      3. 将剪枝比例 reshape 成 (32, 7)，并计算每层整体剪枝率（取均值）；
      4. 利用每层所有模块的 ESD 数值归一化（归一化公式：1 - (esd - esd_min)/(esd_max - esd_min)），
         重要性高（归一化值大）的模块剪枝比例低，重要性低的模块剪枝比例高；
      5. 将每层整体剪枝率按归一化的模块重要性重新分配到每个模块上，返回每层 7 个模块的剪枝比例。
    """
    
    if "opt" in args.model:
        blocks = model.model.decoder.layers    
    else:
        blocks = model.model.layers

    # 假设 find_layers(block) 返回每个 block 中需剪枝的子层字典（顺序一致，每层7个模块）
    layers = [find_layers(block) for block in blocks]
    prunables = []
    for layer in layers:
        for name in layer:
            prunables.append(layer[name].weight.numel())
    layer_num_in_block = int(len(prunables) / len(blocks))  # 应该为7

    # ------------------ ESD 部分 ------------------
    # 加载ESD指标
    metrics = np.load(f"{args.ww_metric_cache}/{args.ww_metric}.npy")
    print("ESD raw metrics:", metrics)
    if args.mapping_type == 'block_wise':
        # 对每个 block 内的 7 个模块取均值
        block_metrics = [np.mean(metrics[i:i+layer_num_in_block])
                         for i in range(0, len(metrics), layer_num_in_block)]
        # 对每个 block 复制 7 次
        metrics = [val for val in block_metrics for _ in range(layer_num_in_block)]
    print("ESD metric values after block_wise processing:", metrics)
            
    scores = torch.tensor(metrics, dtype=torch.float32)
    prunables_tensor = torch.tensor(prunables, dtype=torch.float32)
    max_score = torch.max(scores)
    min_score = torch.min(scores)
    # 线性映射 ESD 数值到 [s1, s2]
    layerwise_pruning_ratios_esd = (((scores - min_score) / (max_score - min_score + eps)) * (s2 - s1) + s1)
    # 校正以满足整体稀疏率要求
    scaler = torch.sum(prunables_tensor) * args.sparsity_ratio / (torch.sum(prunables_tensor * layerwise_pruning_ratios_esd))
    layerwise_pruning_ratios_esd = layerwise_pruning_ratios_esd * scaler
    layerwise_pruning_ratios_esd = layerwise_pruning_ratios_esd.cpu().numpy().tolist()
    print("ESD-based ratios:", layerwise_pruning_ratios_esd)

    # ------------------ 分段重要性部分（代表 gradNorm） ------------------
    # 预定义分段信息，将 32 层划分为 14 个段
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
    # 预定义各段的原始重要性
    seg_importance_raw = np.array([37.25, 10.25, 0.8398, 0.5781, 0.2174, 0.1138,
                                   0.1027, 0.0973, 0.0944, 0.0933, 0.0933, 0.0737,
                                   0.0708, 0.0199], dtype=np.float32)
    eps = 1e-8  # 避免 log(0)
    layer_importance = np.zeros(32, dtype=np.float32)
    for seg_id, layer_list in segments.items():
        for layer_idx in layer_list:
            layer_importance[layer_idx] = seg_importance_raw[seg_id]

    # -------------------------- 再归一化 --------------------------
    # 对赋值后的重要性先进行对数变换，再进行 min-max 归一化到 [0, 1] 范围内
    layer_importance = np.log1p(layer_importance)  # 计算 log(1 + x)
    layer_importance = (layer_importance - layer_importance.min()) / (layer_importance.max() - layer_importance.min() + eps)

    print("Log Scaled Normalized layer importance per layer:")
    print(layer_importance)
    pruning_ratios = 1 - layer_importance

    # 计算当前剪枝比例的均值
    current_mean = np.mean(pruning_ratios)
    target_mean = 0.7  # 目标剪枝比例均值

    # 计算缩放因子，使得最终剪枝比例的均值为 0.7
    scale_factor = target_mean / current_mean

    # 进行缩放
    scaled_pruning_ratios = pruning_ratios * scale_factor

    # 限制范围确保剪枝比例不会超出 [0, 0.99]
    scaled_pruning_ratios = np.clip(scaled_pruning_ratios, 0.0, 0.99)

    # 计算新均值
    new_mean = np.mean(scaled_pruning_ratios)

    # 输出最终剪枝比例
    print("🔥 调整后的剪枝比例:", scaled_pruning_ratios)
    print("🔥 新的均值:", new_mean)
   
    grad_part = np.repeat(scaled_pruning_ratios, layer_num_in_block)
    
    # ------------------ 最终组合 ------------------
    # 最终剪枝比例由 ESD 部分与 grad 部分按权重加权组合（例如：0.8*ESD + 0.2*grad）
    final_pruning_ratios = weight_esd * np.array(layerwise_pruning_ratios_esd) + (1-weight_esd) * grad_part
    print("🔥 最终剪枝比例:", final_pruning_ratios)
    print("all mean: ", np.mean(final_pruning_ratios))
    return final_pruning_ratios
    
def ww_sparsity_llama_rl(args, model, device=torch.device("cuda:0"),
                                s1=0.8, s2=1.2, ratios=None, prune_n=0, prune_m=0,
                                weight_esd=0.95, eps=1e-8):
    if "opt" in args.model:
        blocks = model.model.decoder.layers    
    else:
        blocks = model.model.layers
    layers = [find_layers(blocks)]
    prunables = []
    for layer in layers:
        for name in layer:
            prunables.append(layer[name].weight.numel())
    layer_num_in_block = int(len(prunables) / len(blocks))
    
    # 加载ESD指标
    metrics = np.load(f"{args.ww_metric_cache}/{args.ww_metric}.npy")
    print("ESD raw metrics:", metrics)
    if args.mapping_type == 'block_wise':
        block_metrics = [np.mean(metrics[i:i+layer_num_in_block]) 
                         for i in range(0, len(metrics), layer_num_in_block)]
        metrics = [i for i in block_metrics for j in range(layer_num_in_block)]
    print("ESD metric values after block_wise processing:", metrics)
            
    scores = torch.tensor(metrics, dtype=torch.float32)
    prunables_tensor = torch.tensor(prunables, dtype=torch.float32)
    max_score = torch.max(scores)
    min_score = torch.min(scores)
    # 线性映射到 [s1, s2]
    layerwise_pruning_ratios_esd = (((scores - min_score) / (max_score - min_score)) * (s2 - s1) + s1)
    scaler = torch.sum(prunables_tensor) * args.sparsity_ratio / (torch.sum(prunables_tensor * layerwise_pruning_ratios_esd))
    layerwise_pruning_ratios_esd = layerwise_pruning_ratios_esd * scaler
    layerwise_pruning_ratios_esd = layerwise_pruning_ratios_esd.cpu().numpy().tolist()
    print("ESD-based ratios:", layerwise_pruning_ratios_esd)

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
    print(res)
    
       


    #b = np.array([0.7003036737442017, 0.6655532121658325, 0.7393957376480103, 0.8427770137786865, 0.6547543406486511, 0.7298241853713989, 0.7096095681190491, 0.7544902563095093, 0.6928205490112305, 0.5879737734794617, 0.7208297252655029, 0.7269023656845093, 0.5967667102813721, 0.7670202255249023, 0.7825360298156738, 0.6979405879974365, 0.7198771834373474, 0.717486560344696, 0.6833314299583435, 0.6722944378852844, 0.8652265071868896, 0.6793380379676819, 0.7010731101036072, 0.6536277532577515, 0.6648387908935547, 0.7048466801643372, 0.6144991517066956, 0.7443692088127136, 0.5768176913261414, 0.6605796813964844, 0.6343473792076111, 0.7379485368728638])
    #res = []
    #for i in b:
        #for j in range(7):
            #res.append(i)
    #res= np.array(res)
    #final_pruning_ratios = weight_esd * np.array(layerwise_pruning_ratios_esd) + (1-weight_esd) * res
    #print("🔥 最终剪枝比例:", final_pruning_ratios)
    #print("all mean: ", np.mean(final_pruning_ratios))
    return res

def ww_sparsity_llama2_7b_split(args, model, device=torch.device("cuda:0"),
                                s1=0.8, s2=1.2, ratios=None, prune_n=0, prune_m=0,
                                weight_esd=0.8, eps=1e-8):
    """
    基于 ESD 数值计算 LLaMA 7B 各层（32 层，每层 7 个模块：Q, K, V, Out, Gate, Up, Down）的剪枝比例。
    计算流程：
      1. 加载 ESD 指标，并按照分块（segmentation）计算 block-level 平均 ESD；
      2. 计算线性映射得到每个模块的剪枝比例（总长度 224）；
      3. 将剪枝比例 reshape 成 (32, 7)，并计算每层整体剪枝率（取均值）；
      4. 利用每层所有模块的 ESD 数值归一化（归一化公式：1 - (esd - esd_min)/(esd_max - esd_min)），
         重要性高（归一化值大）的模块剪枝比例低，重要性低的模块剪枝比例高；
      5. 将每层整体剪枝率按归一化的模块重要性重新分配到每个模块上，返回每层 7 个模块的剪枝比例。
    """
    
    if "opt" in args.model:
        blocks = model.model.decoder.layers    
    else:
        blocks = model.model.layers

    # 得到待剪枝层字典，假设 find_layers 返回的顺序与 transformer 层顺序一致，
    # 每个 transformer 层内有7个子层
    layers = [find_layers(blocks)]
    prunables = []
    for layer in layers:
        for name in layer:
            prunables.append(layer[name].weight.numel())
    layer_num_in_block = int(len(prunables) / len(blocks))
    
    # 加载ESD指标
    metrics = np.load(f"{args.ww_metric_cache}/{args.ww_metric}.npy")
    print("ESD raw metrics:", metrics)
    if args.mapping_type == 'block_wise':
        block_metrics = [np.mean(metrics[i:i+layer_num_in_block]) 
                         for i in range(0, len(metrics), layer_num_in_block)]
        metrics = [i for i in block_metrics for j in range(layer_num_in_block)]
    print("ESD metric values after block_wise processing:", metrics)
            
    scores = torch.tensor(metrics, dtype=torch.float32)
    prunables_tensor = torch.tensor(prunables, dtype=torch.float32)
    max_score = torch.max(scores)
    min_score = torch.min(scores)
    # 线性映射到 [s1, s2]
    layerwise_pruning_ratios_esd = (((scores - min_score) / (max_score - min_score)) * (s2 - s1) + s1)
    scaler = torch.sum(prunables_tensor) * args.sparsity_ratio / (torch.sum(prunables_tensor * layerwise_pruning_ratios_esd))
    layerwise_pruning_ratios_esd = layerwise_pruning_ratios_esd * scaler
    layerwise_pruning_ratios_esd = layerwise_pruning_ratios_esd.cpu().numpy().tolist()
    print("ESD-based ratios:", layerwise_pruning_ratios_esd)

    importance = np.array([5.4375, 2.1562, 1.5625, 1.5625, 0.8836, 0.8836, 0.8836, 0.8836, 0.8836, 0.3727,0.3727,0.3727,0.3727,0.3727,0.3727,
        0.2061,0.2061,0.2061,0.1548,0.1548,0.1548,0.1548,0.1237,0.1237,0.1237,0.1138,0.1011,0.1011,0.1016,0.1069,0.0557,0.0415])
    I_min = np.min(importance)
    I_max = np.max(importance)
    norm_importance = (importance - I_min) / (I_max - I_min)
    # 反转：重要性越高（数值大）希望剪枝比例越低
    pre_ratio = 1 - norm_importance
    avg_pre_ratio = np.mean(pre_ratio)
    print("Preliminary importance ratios:", pre_ratio)
    print("Average of importance preliminary ratios:", avg_pre_ratio)
    target_avg = args.sparsity_ratio  # 这里假设 args.sparsity_ratio 代表全局目标剪枝率（例如0.5）
    scale_factor = target_avg / avg_pre_ratio
    final_ratios_importance = pre_ratio * scale_factor
    final_ratios_importance = np.clip(final_ratios_importance, 0.0, 0.99)
    # 扩展：每个 transformer 层内有 layer_num_in_block 子层（例如7个）
    importance_ratios_expanded = []
    for i in final_ratios_importance:
        for j in range(layer_num_in_block):
            importance_ratios_expanded.append(i)
    print("Importance-based expanded ratios:", importance_ratios_expanded)
    
    # ---------------------- 结合两种比例 ----------------------
    # 这里采用加权平均方式，将 ESD-based 和 importance-based 比例融合
    # weight_esd 为权重，默认为0.5，两者各占一半
    if len(layerwise_pruning_ratios_esd) != len(importance_ratios_expanded):
        raise ValueError("Length mismatch between ESD-based and importance-based ratios!")
    
    combined_ratios = []
    for r_esd, r_imp in zip(layerwise_pruning_ratios_esd, importance_ratios_expanded):
        combined = weight_esd * r_esd + (1 - weight_esd) * r_imp
        combined = min(combined, 1.0)
        combined_ratios.append(combined)
    
    print("Combined layerwise pruning ratios:", combined_ratios)
 

    res = []

    for i in range(32):
        #Q
        res.append(combined_ratios[i*7] * 0.148* 7 )
        #K
        res.append(combined_ratios[i*7]  *0.142392 * 7 )
        #V
        res.append(combined_ratios[i*7]  *0.142392  * 7 )
        #OUT
        res.append(combined_ratios[i*7]  * 0.142392 * 7  )
        #GATE1
        res.append(combined_ratios[i*7]  *0.141008  * 7   )
        #UP
        res.append(combined_ratios[i*7]  *0.141808 * 7  )
        #DOWN

        res.append(combined_ratios[i*7]  * 0.142008 * 7  )
    print(res)
    return res

def ww_sparsity_llama3_8b_split(args, model, device=torch.device("cuda:0"),
                                s1=0.8, s2=1.2, ratios=None, prune_n=0, prune_m=0,
                                weight_esd=0.95, eps=1e-8):
    """
    基于 ESD 数值计算 LLaMA 7B 各层（32 层，每层 7 个模块：Q, K, V, Out, Gate, Up, Down）的剪枝比例。
    计算流程：
      1. 加载 ESD 指标，并按照分块（segmentation）计算 block-level 平均 ESD；
      2. 计算线性映射得到每个模块的剪枝比例（总长度 224）；
      3. 将剪枝比例 reshape 成 (32, 7)，并计算每层整体剪枝率（取均值）；
      4. 利用每层所有模块的 ESD 数值归一化（归一化公式：1 - (esd - esd_min)/(esd_max - esd_min)），
         重要性高（归一化值大）的模块剪枝比例低，重要性低的模块剪枝比例高；
      5. 将每层整体剪枝率按归一化的模块重要性重新分配到每个模块上，返回每层 7 个模块的剪枝比例。
    """
    if "opt" in args.model:
        blocks = model.model.decoder.layers    
    else:
        blocks = model.model.layers

    # 得到待剪枝层字典，假设 find_layers 返回的顺序与 transformer 层顺序一致，
    # 每个 transformer 层内有7个子层
    layers = [find_layers(blocks)]
    prunables = []
    for layer in layers:
        for name in layer:
            prunables.append(layer[name].weight.numel())
    layer_num_in_block = int(len(prunables) / len(blocks))
    
    # 加载ESD指标
    metrics = np.load(f"{args.ww_metric_cache}/{args.ww_metric}.npy")
    print("ESD raw metrics:", metrics)
    if args.mapping_type == 'block_wise':
        block_metrics = [np.mean(metrics[i:i+layer_num_in_block]) 
                         for i in range(0, len(metrics), layer_num_in_block)]
        metrics = [i for i in block_metrics for j in range(layer_num_in_block)]
    print("ESD metric values after block_wise processing:", metrics)
            
    scores = torch.tensor(metrics, dtype=torch.float32)
    prunables_tensor = torch.tensor(prunables, dtype=torch.float32)
    max_score = torch.max(scores)
    min_score = torch.min(scores)
    # 线性映射到 [s1, s2]
    layerwise_pruning_ratios_esd = (((scores - min_score) / (max_score - min_score)) * (s2 - s1) + s1)
    scaler = torch.sum(prunables_tensor) * args.sparsity_ratio / (torch.sum(prunables_tensor * layerwise_pruning_ratios_esd))
    layerwise_pruning_ratios_esd = layerwise_pruning_ratios_esd * scaler
    layerwise_pruning_ratios_esd = layerwise_pruning_ratios_esd.cpu().numpy().tolist()
    print("ESD-based ratios:", layerwise_pruning_ratios_esd)
    
    



    res = []

    for i in range(32):
        #Q
        res.append(layerwise_pruning_ratios_esd[i*7] *0.142638 * 7 )
        #K
        res.append(layerwise_pruning_ratios_esd[i*7]  *0.141982 * 7 )
        #V
        res.append(layerwise_pruning_ratios_esd[i*7]  * 0.148107 * 7 )
        #OUT
        res.append(layerwise_pruning_ratios_esd[i*7]  * 0.142638 * 7  )
        #GATE1
        res.append(layerwise_pruning_ratios_esd[i*7]  *0.142795  * 7   )
        #UP
        res.append(layerwise_pruning_ratios_esd[i*7]  *0.142795 * 7  )
        #DOWN

        res.append(layerwise_pruning_ratios_esd[i*7]  * 0.142795 * 7  )
    print(res)
    return res
    

def ww_sparsity_llama3_8b(args, model, device=torch.device("cuda:0"),
                         s1=0.8, s2=1.2, ratios=None, prune_n=0, prune_m=0,
                         weight_esd=0.5, eps=1e-8):
    if "opt" in args.model:
        blocks = model.model.decoder.layers    
    else:
        blocks = model.model.layers

    # 得到待剪枝层字典，假设 find_layers 返回的顺序与 transformer 层顺序一致，
    # 每个 transformer 层内有7个子层
    layers = [find_layers(blocks)]
    prunables = []
    for layer in layers:
        for name in layer:
            prunables.append(layer[name].weight.numel())
    layer_num_in_block = int(len(prunables) / len(blocks))
    
    # 加载ESD指标
    metrics = np.load(f"{args.ww_metric_cache}/{args.ww_metric}.npy")
    print("ESD raw metrics:", metrics)
    if args.mapping_type == 'block_wise':
        block_metrics = [np.mean(metrics[i:i+layer_num_in_block]) 
                         for i in range(0, len(metrics), layer_num_in_block)]
        metrics = [i for i in block_metrics for j in range(layer_num_in_block)]
    print("ESD metric values after block_wise processing:", metrics)
            
    scores = torch.tensor(metrics, dtype=torch.float32)
    prunables_tensor = torch.tensor(prunables, dtype=torch.float32)
    max_score = torch.max(scores)
    min_score = torch.min(scores)
    # 线性映射到 [s1, s2]
    layerwise_pruning_ratios_esd = (((scores - min_score) / (max_score - min_score)) * (s2 - s1) + s1)
    scaler = torch.sum(prunables_tensor) * args.sparsity_ratio / (torch.sum(prunables_tensor * layerwise_pruning_ratios_esd))
    layerwise_pruning_ratios_esd = layerwise_pruning_ratios_esd * scaler
    layerwise_pruning_ratios_esd = layerwise_pruning_ratios_esd.cpu().numpy().tolist()
    print("ESD-based ratios:", layerwise_pruning_ratios_esd)
    
    segments = {
        0: [0],
        1: [1, 2, 3, 4, 5, 6],
        2: [7, 8, 9, 10, 11, 12],
        3: [13, 14],
        4: [15],
        5: [16, 17, 18],
        6: [19],
        7: [20],
        8: [21, 22, 23],
        9: [24, 25, 26, 27],
        10: [28],
        11: [29],
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
    print(res)
    return res
 

def ww_sparsity_llama_7b(args, model, device=torch.device("cuda:0"),
                         s1=0.8, s2=1.2, ratios=None, prune_n=0, prune_m=0,
                         weight_esd=0.8, eps=1e-8):
    if "opt" in args.model:
        blocks = model.model.decoder.layers    
    else:
        blocks = model.model.layers

    # 得到待剪枝层字典，假设 find_layers 返回的顺序与 transformer 层顺序一致，
    # 每个 transformer 层内有7个子层
    layers = [find_layers(blocks)]
    prunables = []
    for layer in layers:
        for name in layer:
            prunables.append(layer[name].weight.numel())
    layer_num_in_block = int(len(prunables) / len(blocks))
    
    # 加载ESD指标
    metrics = np.load(f"{args.ww_metric_cache}/{args.ww_metric}.npy")
    print("ESD raw metrics:", metrics)
    if args.mapping_type == 'block_wise':
        block_metrics = [np.mean(metrics[i:i+layer_num_in_block]) 
                         for i in range(0, len(metrics), layer_num_in_block)]
        metrics = [i for i in block_metrics for j in range(layer_num_in_block)]
    print("ESD metric values after block_wise processing:", metrics)
        
    scores = torch.tensor(metrics, dtype=torch.float32)
    prunables_tensor = torch.tensor(prunables, dtype=torch.float32)
    max_score = torch.max(scores)
    min_score = torch.min(scores)
    # 线性映射到 [s1, s2]
    layerwise_pruning_ratios_esd = (((scores - min_score) / (max_score - min_score)) * (s2 - s1) + s1)
    scaler = torch.sum(prunables_tensor) * args.sparsity_ratio / (torch.sum(prunables_tensor * layerwise_pruning_ratios_esd))
    layerwise_pruning_ratios_esd = layerwise_pruning_ratios_esd * scaler
    layerwise_pruning_ratios_esd = layerwise_pruning_ratios_esd.cpu().numpy().tolist()
    print("ESD-based ratios:", layerwise_pruning_ratios_esd)
    
    importance = np.array([0.3262, 0.2539,0.1846, 0.1846,0.0899,0.0899,0.0899,0.0899,0.0899,0.0899,0.0899,0.0481,0.0481,
                       0.0389,0.0389,0.0389,0.0317,0.0268,0.0268,0.0268,0.0227,0.0191,0.0191,0.0191,0.0191,
                       0.0191,0.0191,0.0191,0.0164,0.0157,0.0154,0.0086])
    I_min = np.min(importance)
    I_max = np.max(importance)
    norm_importance = (importance - I_min) / (I_max - I_min)
    # 反转：重要性越高（数值大）希望剪枝比例越低
    pre_ratio = 1 - norm_importance
    avg_pre_ratio = np.mean(pre_ratio)
    print("Preliminary importance ratios:", pre_ratio)
    print("Average of importance preliminary ratios:", avg_pre_ratio)
    target_avg = args.sparsity_ratio  # 这里假设 args.sparsity_ratio 代表全局目标剪枝率（例如0.5）
    scale_factor = target_avg / avg_pre_ratio
    final_ratios_importance = pre_ratio * scale_factor
    final_ratios_importance = np.clip(final_ratios_importance, 0.0, 0.99)
    # 扩展：每个 transformer 层内有 layer_num_in_block 子层（例如7个）
    importance_ratios_expanded = []
    for i in final_ratios_importance:
        for j in range(layer_num_in_block):
            importance_ratios_expanded.append(i)
    print("Importance-based expanded ratios:", importance_ratios_expanded)
    
    # ---------------------- 结合两种比例 ----------------------
    # 这里采用加权平均方式，将 ESD-based 和 importance-based 比例融合
    # weight_esd 为权重，默认为0.5，两者各占一半
    if len(layerwise_pruning_ratios_esd) != len(importance_ratios_expanded):
        raise ValueError("Length mismatch between ESD-based and importance-based ratios!")
    
    combined_ratios = []
    for r_esd, r_imp in zip(layerwise_pruning_ratios_esd, importance_ratios_expanded):
        combined = weight_esd * r_esd + (1 - weight_esd) * r_imp
        combined = min(combined, 1.0)
        combined_ratios.append(combined)
    
    print("Combined layerwise pruning ratios:", combined_ratios)
    return combined_ratios

def ww_sparsity_test_3b(args, model, device=torch.device("cuda:0"),
                         s1=0.8, s2=1.2, ratios=None, prune_n=0, prune_m=0,
                         weight_esd=0.8, eps=1e-8):
    
    if "opt" in args.model:
        blocks = model.model.decoder.layers    
    else:
        blocks = model.model.layers

    # 得到待剪枝层字典，假设 find_layers 返回的顺序与 transformer 层顺序一致，
    # 每个 transformer 层内有7个子层
    layers = [find_layers(blocks)]
    prunables = []
    for layer in layers:
        for name in layer:
            prunables.append(layer[name].weight.numel())
    layer_num_in_block = int(len(prunables) / len(blocks))
    
    # 加载ESD指标
    metrics = np.load(f"{args.ww_metric_cache}/{args.ww_metric}.npy")
    #print("ESD raw metrics:", metrics)
    
    #print("ESD metric values after block_wise processing:", metrics)
    new = []
    for  i in range(32):
        val = 0
        val += metrics[i*7 ] + metrics[i*7+3] + metrics[i*7+1]
        val /= 3
        for j in range(7):
            new.append(val)
    metrics = np.array(new)

    scores = torch.tensor(metrics, dtype=torch.float32)
    prunables_tensor = torch.tensor(prunables, dtype=torch.float32)
    max_score = torch.max(scores)
    min_score = torch.min(scores)
    # 线性映射到 [s1, s2]
    layerwise_pruning_ratios_esd = (((scores - min_score) / (max_score - min_score)) * (s2 - s1) + s1)
    scaler = torch.sum(prunables_tensor) * args.sparsity_ratio / (torch.sum(prunables_tensor * layerwise_pruning_ratios_esd))
    layerwise_pruning_ratios_esd = layerwise_pruning_ratios_esd * scaler
    layerwise_pruning_ratios_esd = layerwise_pruning_ratios_esd.cpu().numpy().tolist()
    print("ESD-based ratios:", layerwise_pruning_ratios_esd)
    return layerwise_pruning_ratios_esd
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
    print(res)
    return res
    a = []
    for i in c:
        for j in range(7):
            a.append(i)
    return layerwise_pruning_ratios_esd
    if "opt" in args.model:
        blocks = model.model.decoder.layers    
    else:
        blocks = model.model.layers

    # 得到待剪枝层字典，假设 find_layers 返回的顺序与 transformer 层顺序一致，
    # 每个 transformer 层内有7个子层
    layers = [find_layers(blocks)]
    prunables = []
    for layer in layers:
        for name in layer:
            prunables.append(layer[name].weight.numel())
    layer_num_in_block = int(len(prunables) / len(blocks))
    
    # 加载ESD指标
    metrics = np.load(f"{args.ww_metric_cache}/{args.ww_metric}.npy")
    print("ESD raw metrics:", metrics)
    if args.mapping_type == 'block_wise':
        block_metrics = [np.mean(metrics[i:i+layer_num_in_block]) 
                         for i in range(0, len(metrics), layer_num_in_block)]
        metrics = [i for i in block_metrics for j in range(layer_num_in_block)]
    print("ESD metric values after block_wise processing:", metrics)
            
    scores = torch.tensor(metrics, dtype=torch.float32)
    prunables_tensor = torch.tensor(prunables, dtype=torch.float32)
    max_score = torch.max(scores)
    min_score = torch.min(scores)
    # 线性映射到 [s1, s2]
    layerwise_pruning_ratios_esd = (((scores - min_score) / (max_score - min_score)) * (s2 - s1) + s1)
    scaler = torch.sum(prunables_tensor) * args.sparsity_ratio / (torch.sum(prunables_tensor * layerwise_pruning_ratios_esd))
    layerwise_pruning_ratios_esd = layerwise_pruning_ratios_esd * scaler
    layerwise_pruning_ratios_esd = layerwise_pruning_ratios_esd.cpu().numpy().tolist()
    print("ESD-based ratios:", layerwise_pruning_ratios_esd)

    for i in range(32):
        c.append(layerwise_pruning_ratios_esd[i*7])
    print(c)
    return layerwise_pruning_ratios_esd
#########################################################################################################################

def prune_magnitude(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0, ratios=None):
    if "OPT" in model.__class__.__name__:
        layers = model.model.decoder.layers
    else:    
        layers = model.model.layers
    
    layer_num = len(find_layers(layers))
    if ratios is None:
        ratios = [args.sparsity_ratio for i in range(layer_num)]

    k=0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        for name in subset:
            W = subset[name].weight.data 
            W_metric = torch.abs(W)
            print(ratios[k])
            thresh = torch.sort(W_metric.flatten().cuda())[0][int(W.numel()*ratios[k])].cpu()
            W_mask = (W_metric<=thresh)
            k+=1

            W[W_mask] = 0


def prune_wanda(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0, ratios=None):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    print("loading calibdation data")
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        if "OPT" in model.__class__.__name__:
            inps, outs, attention_mask, position_ids = prepare_calibration_input_opt(model, dataloader, device)
        else:
            inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)


    print ("inps",inps)

    if "OPT" in model.__class__.__name__:
        layers = model.model.decoder.layers
    else:    
        layers = model.model.layers
    
    layer_num = len(find_layers(layers))
    if ratios is None:
        ratios = [args.sparsity_ratio for i in range(layer_num)]
    
    k=0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        if f"model.layers.{i}" in model.hf_device_map:   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            print("using wanda!")
            dev = model.hf_device_map[f"model.layers.{i}"]
            print(f"layer {i} device {dev}")
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in subset:
            print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))

            W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
            if prune_n != 0:
                # structured n:m sparsity
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)

                if args.use_variant:
                    # wanda variant 
                    tmp_metric = torch.cumsum(sort_res[0], dim=1)
                    sum_before = W_metric.sum(dim=1)

                    alpha = 0.4
                    alpha_hist = [0., 0.8]
                    W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    while (torch.abs(cur_sparsity - args.sparsity_ratio)>0.001) and (alpha_hist[1]-alpha_hist[0]>=0.001):
                        if cur_sparsity > args.sparsity_ratio:
                            alpha_new = (alpha + alpha_hist[0]) / 2.0
                            alpha_hist[1] = alpha
                        else:
                            alpha_new = (alpha + alpha_hist[1]) / 2.0
                            alpha_hist[0] = alpha

                        alpha = alpha_new 
                        W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    print(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
                else:
                    # unstructured pruning
                    indices = sort_res[1][:,:int(W_metric.shape[1]*ratios[k])]
                    k+=1
                    W_mask.scatter_(1, indices, True)
#             print ("W_mask",W_mask)
            subset[name].weight.data[W_mask] = 0  ## set weights to zero 

        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        inps, outs = outs, inps

    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()


@torch.no_grad()
def prune_sparsegpt(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0, ratios=None):
    ## SparseGPT code available at: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa
    print('Starting ...')
    use_cache = model.config.use_cache
    model.config.use_cache = False
    
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
    
    with torch.no_grad():
        if "OPT" in model.__class__.__name__:
            inps, outs, attention_mask, position_ids = prepare_calibration_input_opt(model, dataloader, device)
        else:
            inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)

    if "OPT" in model.__class__.__name__:
        layers = model.model.decoder.layers
    else:    
        layers = model.model.layers        
    
    
    layer_num = len(find_layers(layers))
    if ratios is None:
        ratios = [args.sparsity_ratio for i in range(layer_num)]
    
    print('Ready.')
    k=0
    for i in range(len(layers)):
        layer = layers[i]
        if f"model.layers.{i}" in model.hf_device_map:
            print("using !")
            dev = model.hf_device_map[f"model.layers.{i}"]
            print(f"layer {i} device {dev}")
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        subset = find_layers(layer)

        gpts = {}
        for name in subset:
            gpts[name] = SparseGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            if "OPT" in model.__class__.__name__:
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
            else:
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in gpts:
            print(i, name)
            print('Pruning ...')

            gpts[name].fasterprune(ratios[k], prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=128)
            gpts[name].free()
            k+=1

        for j in range(args.nsamples):
            if "OPT" in model.__class__.__name__:
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
            else:
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        layers[i] = layer 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()

###############################################################################################################
def prune_magnitude_ww(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    s1 = 1.0 - args.epsilon
    s2 = 1.0 + args.epsilon
    
    all_layer_ratio = ww_sparsity(args, model, device, s1, s2)
    # magnitude pruning
    prune_magnitude(args, model, tokenizer, device, ratios=all_layer_ratio)

def prune_magnitude_ww2(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    s1 = 1.0 - args.epsilon
    s2 = 1.0 + args.epsilon
    
    all_layer_ratio = ww_sparsity_test_3b(args, model, device, s1, s2)
    # magnitude pruning
    prune_magnitude(args, model, tokenizer, device, ratios=all_layer_ratio)
    

def prune_wanda_ww(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    s1 = 1.0 - args.epsilon
    s2 = 1.0 + args.epsilon

    all_layer_ratio = ww_sparsity(args, model, device, s1, s2)
    # wanda pruning
    prune_wanda(args, model, tokenizer, device, ratios=all_layer_ratio)

def prune_wanda_ww2(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    s1 = 1.0 - args.epsilon
    s2 = 1.0 + args.epsilon

    all_layer_ratio = ww_sparsity_test_3b(args, model, device, s1, s2)
    # wanda pruning
    prune_wanda(args, model, tokenizer, device, ratios=all_layer_ratio)   
    
def prune_sparsegpt_ww(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    s1 = 1.0 - args.epsilon
    s2 = 1.0 + args.epsilon

    all_layer_ratio = ww_sparsity(args, model, device, s1, s2)
    # sparsegpt pruning
    prune_sparsegpt(args, model, tokenizer, device, ratios=all_layer_ratio)

def prune_sparsegpt_ww2(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    s1 = 1.0 - args.epsilon
    s2 = 1.0 + args.epsilon

    all_layer_ratio = ww_sparsity_test_3b(args, model, device, s1, s2)
    # sparsegpt pruning
    prune_sparsegpt(args, model, tokenizer, device, ratios=all_layer_ratio)