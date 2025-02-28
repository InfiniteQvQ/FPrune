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

def entropy_weighted_aggregate(x, eps=1e-8):
    # 将 x 转为 numpy 数组
    x = np.array(x)
    total = np.sum(x) + eps
    # 归一化
    p = x / total
    # 计算每个分量的“信息量”，这里用 1 - p * log(p+eps)
    terms = 1 - p * np.log(p + eps)
    # 归一化权重
    weights = terms / (np.sum(terms) + eps)
    # 综合指标为各值的加权和
    composite = np.sum(weights * x)
    return composite

def entropy_weighted_composite(x, eps=1e-8):
    """
    对于一个 transformer 层内的多个 ESD 数值 x（例如长度为7的数组），
    采用熵加权方法计算综合指标：
      1. 归一化： p_i = x_i / (sum(x) + eps)
      2. 计算权重： w_i = 1 - p_i * log(p_i + eps)
      3. 归一化权重： w_i = w_i / (sum(w) + eps)
      4. 综合指标： composite = sum(w_i * x_i)
    """
    x = np.array(x)
    total = np.sum(x) + eps
    p = x / total
    weights = 1 - p * np.log(p + eps)
    weights = weights / (np.sum(weights) + eps)
    composite = np.sum(weights * x)
    return composite

def ww_sparsity_enhanced(args, model, device=torch.device("cuda:0"),
                           s1=0.8, s2=1.2, ratios=None, prune_n=0, prune_m=0, eps=1e-8):
    # 定义模型分块（例如，每个 transformer 层由7个子矩阵构成）
    SEGMENTS = [
        {"layers": [0], "repeat": 7},
        {"layers": [1], "repeat": 7},
        {"layers": [2], "repeat": 7},
        {"layers": [3,4,5,6], "repeat": 7},
        {"layers": [7,8,9,10,11,12], "repeat": 7},
        {"layers": [13], "repeat": 7},
        {"layers": [14], "repeat": 7},
        {"layers": [15,16], "repeat": 7},
        {"layers": [17,18], "repeat": 7},
        {"layers": [19,20,21,22], "repeat": 7},
        {"layers": [23,24], "repeat": 7},
        {"layers": [25], "repeat": 7},
        {"layers": [26], "repeat": 7},
        {"layers": [27], "repeat": 7}
    ]
    
    # 根据模型名称获取相应的 block 层
    if "opt" in args.model:
        blocks = model.model.decoder.layers    
    else:
        blocks = model.model.layers

    # 使用 find_layers（你需要自行实现）获得待剪枝层字典，字典顺序应与 transformer 层顺序一致，
    # 每个 transformer 层内部有 7 个子矩阵
    layers_dict = find_layers(blocks)
    # 计算所有子层的参数数量
    all_prunables = []
    for name in layers_dict:
        all_prunables.append(layers_dict[name].weight.numel())
    total_sub_layers = len(all_prunables)
    print("Total prunable sub-layers:", total_sub_layers)

    # 假设每个 transformer 层有7个子层
    layer_num_in_block = 7
    num_transformer = int(total_sub_layers / layer_num_in_block)
    print("Transformer layer count:", num_transformer)

    # 加载原始的 ESD 指标，长度应为 num_transformer * layer_num_in_block
    raw_metrics = np.load(f"{args.ww_metric_cache}/{args.ww_metric}.npy")
    print("Loaded raw metrics:", raw_metrics)

    # 对于每个 transformer 层（连续的7个数值），用熵加权方法计算一个综合指标
    composite_list = []
    for i in range(num_transformer):
        start = i * layer_num_in_block
        end = start + layer_num_in_block
        sub_vals = raw_metrics[start:end]
        comp = entropy_weighted_composite(sub_vals, eps)
        composite_list.append(comp)
    print("Composite per transformer layer:", composite_list)

    # 将每个 transformer 层的综合指标复制7次，扩展为每个子层都有相同的数值
    expanded_metrics = [val for val in composite_list for _ in range(layer_num_in_block)]
    print("Expanded metric values (entropy weighted composite):", expanded_metrics)

    # -------------------- 分块全局映射 --------------------
    # 对于每个 SEGMENT，根据定义计算加权平均指标和总参数数（利用所有子层对应的参数数量）
    seg_metrics = []      # 每个分块的加权平均指标
    seg_prunables = []    # 每个分块的总参数数量
    for seg in SEGMENTS:
        expanded_layers = []
        for l in seg["layers"]:
            # 对于该大层，重复 seg["repeat"] 次
            expanded_layers.extend(range(l * seg["repeat"], (l + 1) * seg["repeat"]))
        total_params = sum(all_prunables[i] for i in expanded_layers)
        weighted_metric = sum(expanded_metrics[i] * all_prunables[i] for i in expanded_layers) / total_params
        seg_metrics.append(weighted_metric)
        seg_prunables.append(total_params)
    
    scores = torch.tensor(seg_metrics, dtype=torch.float32)
    prunables_tensor = torch.tensor(seg_prunables, dtype=torch.float32)
    print("Segment prunable parameters:", prunables_tensor)

    # 将每个分块的加权平均指标映射到 [s1, s2] 区间（线性映射）
    max_score = torch.max(scores)
    min_score = torch.min(scores)
    seg_ratios = ((scores - min_score) / (max_score - min_score)) * (s2 - s1) + s1

    # 全局校准，使各分块的剪枝数量之和满足全局稀疏率
    total_prunable = torch.sum(prunables_tensor)
    target_pruned = total_prunable * args.sparsity_ratio
    scaler = target_pruned / torch.sum(prunables_tensor * seg_ratios)
    calibrated_seg_ratios = seg_ratios * scaler

    # -------------------- 分块内细粒度调整 --------------------
    # 对于每个分块，我们希望该分块内各子层的平均剪枝比例为 R_block，同时根据子层原始指标做微调
    final_layer_ratios = []
    for seg_idx, seg in enumerate(SEGMENTS):
        R_block = calibrated_seg_ratios[seg_idx].item()
        expanded_layers = []
        for l in seg["layers"]:
            expanded_layers.extend(range(l * seg["repeat"], (l + 1) * seg["repeat"]))
        # 提取该分块内各子层的指标
        block_metrics = [expanded_metrics[i] for i in expanded_layers]
        # 利用一个简单映射函数 F(metric)=1/(metric+eps)
        f_values = [1.0 / (m + eps) for m in block_metrics]
        avg_f = sum(f_values) / len(f_values)
        for f in f_values:
            ratio = R_block * (f / avg_f)
            final_layer_ratios.append(ratio)
    
    final_layer_ratios = torch.tensor(final_layer_ratios, dtype=torch.float32)
    final_layer_ratios = torch.clamp(final_layer_ratios, max=1.0)
    final_ratios = final_layer_ratios.cpu().numpy().tolist()
    print("Final layerwise pruning ratios (clipped to 1.0):", final_ratios)
    print(f"Generated {len(final_ratios)} ratios")
    
    return final_ratios
    
def ww_sparsity_test(args, model, device=torch.device("cuda:0"),
                         s1=0.8, s2=1.2, ratios=None, prune_n=0, prune_m=0,
                         weight_esd=0.1, eps=1e-8):
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
    
    # ---------------------- 基于 importance 的剪枝比例计算 ----------------------
    # 这里 importance 数值（通常来自 gradNorm 或其它评价）是针对 transformer 层的
    # 假设 importance 数组长度等于 transformer 层数
    importance = np.array([4.0938, 3.2188, 2.5, 1.9590, 1.9590, 1.9590, 1.9590, 
                           1.6719, 1.6719, 1.6719, 1.6719, 1.6719, 1.6719, 1.4766, 
                           1.3438, 1.2461, 1.2461, 1.0820, 1.0820, 0.8174, 0.8174, 
                           0.8174, 0.8174, 0.5762, 0.5762, 0.4531, 0.1699, 0.0952])
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
    return combined_ratios

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
    
    # Step 1: 获取 Transformer 层以及各模块参数数量
    if "opt" in args.model:
        blocks = model.model.decoder.layers    
    else:
        blocks = model.model.layers

    # 假设 find_layers 返回每层的字典，字典中包含各模块的权重
    layers = [find_layers(blocks)]
    prunables = []
    for layer in layers:
        for name in layer:
            prunables.append(layer[name].weight.numel())
    # 每层模块数
    layer_num_in_block = int(len(prunables) / len(blocks))
    
    # Step 2: 加载 ESD 指标并分块处理
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
    fisher_scores_flat = np.array([
        3.109375, 2.5625, 39.46875, 59.4375, 31.40625, 35.09375, 62.9375,
        6.4765625, 6.58203125, 52.84375, 51.0625, 65.3125, 76.8125, 101.6875,
        10.734375, 9.671875, 55.25, 53.8125, 83.625, 98.375, 124.4375,
        11.609375, 9.0703125, 25.3125, 28.25, 92.875, 100.3125, 106.4375,
        11.921875, 9.34375, 22.75, 26.34375, 94.6875, 102.625, 100.75,
        13.5625, 11.53125, 21.4375, 24.578125, 84.8125, 95.875, 88.625,
        10.8359375, 9.109375, 20.375, 25.140625, 75.0625, 82.25, 75.125,
        9.0078125, 7.94140625, 18.71875, 24.078125, 61.71875, 67.3125, 64.6875,
        8.765625, 7.53125, 18.90625, 26.046875, 47.59375, 53.03125, 56.84375,
        7.66796875, 6.48828125, 14.5546875, 20.125, 42.28125, 47.40625, 52.46875,
        8.265625, 7.27734375, 17.703125, 22.25, 41.75, 46.1875, 51.3125,
        8.96875, 7.66796875, 20.375, 25.453125, 40.65625, 44.96875, 52.15625,
        8.28125, 6.8828125, 17.6875, 22.09375, 41.4375, 46.84375, 53.46875,
        8.7109375, 7.34375, 18.03125, 21.125, 42.125, 46.6875, 53.65625,
        9.0859375, 8.2734375, 21.59375, 23.734375, 44.3125, 49.34375, 54.75,
        8.5546875, 7.35546875, 19.6875, 21.90625, 45.25, 49.59375, 56.6875,
        9.421875, 8.375, 23.71875, 25.015625, 47.28125, 52.0, 62.125,
        9.265625, 7.6953125, 20.125, 23.140625, 48.6875, 52.96875, 64.0,
        7.609375, 6.54296875, 17.8125, 19.5625, 50.03125, 53.53125, 65.25,
        7.75390625, 6.72265625, 16.671875, 18.171875, 52.125, 55.125, 67.6875,
        9.84375, 9.0625, 21.203125, 22.890625, 53.09375, 55.6875, 68.875,
        7.42578125, 6.45703125, 13.796875, 15.390625, 53.53125, 55.5625, 67.9375,
        5.5625, 5.671875, 14.9609375, 15.0703125, 55.0625, 56.09375, 66.9375,
        6.05859375, 5.75, 15.859375, 16.3125, 55.625, 55.875, 66.0625,
        5.2734375, 5.109375, 12.28125, 12.546875, 56.46875, 57.21875, 66.1875,
        7.734375, 7.640625, 16.71875, 16.984375, 56.875, 56.78125, 64.5,
        5.53125, 5.56640625, 15.390625, 14.8984375, 58.28125, 57.9375, 64.4375,
        6.88671875, 7.74609375, 17.90625, 17.359375, 60.15625, 59.53125, 65.5625,
        8.1171875, 8.96875, 20.453125, 20.984375, 63.1875, 62.125, 70.5,
        7.60546875, 7.23046875, 17.28125, 16.53125, 70.875, 71.5625, 80.375,
        7.4765625, 6.92578125, 18.078125, 18.15625, 82.3125, 83.5, 124.0625,
        16.375, 13.75, 41.71875, 33.21875, 95.9375, 103.375, 132.625
    ])
    num_layers = 32
    num_modules = 7
   # fisher_scores = fisher_scores_flat.reshape(num_layers, num_modules)
    ep = 1e-8
    module_prune_allocations = []  # 存储每层内部各模块的剪枝比例分配
    s1 = 0.9
    s2 = 1
    metrics = np.load(f"{args.ww_metric_cache}/{args.ww_metric}.npy")
    esd_scores = metrics.reshape(num_layers, num_modules)  # 重新整理成 (num_layers, num_modules) 形状

    for i in range(num_layers):
        # 取出第 i 层 7 个模块的 ESD 分数
        layer_esd = esd_scores[i]  # shape: (7,)

        # **第一步**：对 ESD 进行线性映射到 [s1, s2]
        min_f = layer_esd.min()
        max_f = layer_esd.max()
        mapped_esd = ((layer_esd - min_f) / (max_f - min_f + ep)) * (s2 - s1) + s1
        # 这里的 mapped_esd 越大，表示剪枝比例越大

        # **第二步**：归一化，使得 7 个模块的权重之和为 1
        norm_mapped = mapped_esd / mapped_esd.sum()

        # **第三步**：保持该层剪枝比例均值 = `layerwise_pruning_ratios_esd[i]`
        # `layerwise_pruning_ratios_esd[i*7]` 代表该层的目标剪枝比例（例如 0.7）
        target_prune_ratio = layerwise_pruning_ratios_esd[i * 7]  # 该层的目标剪枝比例
        module_prune_ratio = norm_mapped * (num_modules * target_prune_ratio)  # 使得均值等于 target_prune_ratio

        module_prune_allocations.append(module_prune_ratio)

    # **最终转换为 1D 数组**
    module_prune_allocations = np.array(module_prune_allocations).flatten()  # 变成 1D 数组

    # **打印检查**
    print("Flattened module-wise pruning allocations:", module_prune_allocations)
    return module_prune_allocations
    


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

    importance = np.array([16.8750,8.9688,8.9688,8.9688,8.9688,8.9688,8.9688,3.8203,3.8203,3.8203,3.8203,3.8203,3.8203,2.5078,2.5078,
                       2.2188, 1.7943,1.7943,1.7943,1.5469,1.4453,1.2474,1.2474,1.2474, 0.7959,0.7959,0.7959,0.7959,0.5078,0.3516,0.1592,0.1187])
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

def ww_sparsity_llama_7b(args, model, device=torch.device("cuda:0"),
                         s1=0.8, s2=1.2, ratios=None, prune_n=0, prune_m=0,
                         weight_esd=0.2, eps=1e-8):
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

def ww_sparsity_test_uni(args, model, device=torch.device("cuda:0"),
                         s1=0.8, s2=1.2, ratios=None, prune_n=0, prune_m=0,
                         weight_esd=0.5, eps=1e-8):
    a = []
    for i in range(32 * 7):
        a.append(args.sparsity_ratio)
    a = torch.tensor(a, dtype=torch.float32)
    print(a)
    return  a.cpu().numpy().tolist()
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
    
    all_layer_ratio = ww_sparsity_llama2_7b_split(args, model, device, s1, s2)
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

    all_layer_ratio = ww_sparsity_llama2_7b_split(args, model, device, s1, s2)
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

    all_layer_ratio = ww_sparsity_llama2_7b_split(args, model, device, s1, s2)
    # sparsegpt pruning
    prune_sparsegpt(args, model, tokenizer, device, ratios=all_layer_ratio)