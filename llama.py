import time 
from tqdm import tqdm
import torch 
import torch.nn as nn 
from .sparsegpt import SparseGPT 
from .wanda import WandaWrappedGPT
from .wandd import UWrappedGPT
from .blockgpt import LlamaAttentionGPT, LlamaMLPGPT
from .data import get_loaders 
import matplotlib.pyplot as plt
import os
import numpy as np 
DEBUG = False

class WrappedGPT:
    """
    This class wraps a GPT layer for specific operations.
    """

    def __init__(self, layer, layer_id=0, layer_name="none"):
        self.layer = layer
        self.dev = self.layer.weight.device
        self.rows = layer.weight.data.shape[0]
        self.columns = layer.weight.data.shape[1]

        self.scaler_row = torch.zeros((self.columns), device=self.dev)
        self.nsamples = 0

        self.layer_id = layer_id 
        self.layer_name = layer_name

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()

        self.scaler_row *= self.nsamples / (self.nsamples+tmp)
        self.nsamples += tmp

        inp = inp.type(torch.float32)
        self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2  / self.nsamples

def prepare_calibration_input_opt(model, dataloader, device):
    layers = model.model.decoder.layers
    
    if "model.embed_tokens" in model.hf_device_map:
        device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((16, model.seqlen, model.config.hidden_size), dtype=dtype, device=device)
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

def prepare_calibration_input(model, dataloader, nsample, device):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    if "model.embed_tokens" in model.hf_device_map:
        device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((nsample, model.seqlen, model.config.hidden_size), dtype=dtype, device=device)
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

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    model.config.use_cache = use_cache

    return inps, outs, attention_mask, position_ids 

def return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before):
    thres_cumsum = sum_before * alpha 
    sort_mask = tmp_metric <= thres_cumsum.reshape((-1,1))
    thres = torch.gather(sort_res[0], dim=1, index=sort_mask.sum(dim=1, keepdims=True)-1)
    W_mask = (W_metric <= thres)
    cur_sparsity = (W_mask==True).sum() / W_mask.numel()
    return W_mask, cur_sparsity

def prune_magnitude(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    layers = model.model.layers 

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        for name in subset:
            W = subset[name].weight.data 
            W_metric = torch.abs(W)
            if prune_n != 0:
                W_mask = (torch.zeros_like(W)==1)
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                thresh = torch.sort(W_metric.flatten().cuda())[0][int(W.numel()*args.sparsity_ratio)].cpu()
                W_mask = (W_metric<=thresh)

            W[W_mask] = 0

def prune_wanda(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0, ratios=None):
    res = [0.5293006074418442, 0.6358363452242717, 0.7748192144399018, 0.7234297956976186, 0.705076419482818, 0.7139799622267773, 0.747167659368869, 0.5980377426491537, 0.5390139233026938, 0.5062957468510644, 0.5670330803422492, 0.5246790027265411, 0.5556295724845334, 0.5940288875974529, 0.5995001962463322, 0.672528180351988, 0.7913333400142103, 0.7112298065103565, 0.6983272207225116, 0.7847523906736767, 0.7486794849375509, 0.7650636317350729, 0.8405095219304548, 0.8392653829077255, 0.8146051974336859, 0.7603679049430712, 0.7766239792175328, 0.8103622440266844, 0.9402635298662624, 0.7872863009674198, 0.6663283535177681, 0.6786439656583021]
    r = []
    for i in res:
        for j in range(7):
            r.append(i)
    ratios = np.array(r)
    use_cache = model.config.use_cache 
    model.config.use_cache = False 
    
    print("loading calibdation data")
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        if "OPT" in model.__class__.__name__:
            inps, outs, attention_mask, position_ids = prepare_calibration_input_opt(model, dataloader, device)
        else:
            inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, 16, device)


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

def prune_wandd(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    print("loading calibdation data")
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, args.nsamples, device)
    
    layers = model.model.layers
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        if f"model.layers.{i}" in model.hf_device_map:   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = UWrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in subset:
            print(f"pruning layer {i} name {name}")
            print(subset[name].weight)
            print(wrapped_layers[name].scaler)
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler)
            print(W_metric)
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
                    indices = sort_res[1][:,:int(W_metric.shape[1]*args.sparsity_ratio)]
                    W_mask.scatter_(1, indices, True)

            subset[name].weight.data[W_mask] = 0  ## set weights to zero 

        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        inps, outs = outs, inps

    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()
    
@torch.no_grad()
def prune_sparsegpt(args, model, tokenizer, dev, prune_n=0, prune_m=0):
    ## SparseGPT code available at: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa
    print('Starting ...')
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers
    start_time = time.time()

    if "model.embed_tokens" in model.hf_device_map:
        dev = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
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
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    print('Ready.')

    for i in range(len(layers)):
        layer = layers[i]
        if f"model.layers.{i}" in model.hf_device_map:
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
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in gpts:
            print(i, name)
            print('Pruning ...')

            gpts[name].fasterprune(args.sparsity_ratio, prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=128)
            gpts[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        layers[i] = layer 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()
    end_time = time.time()
    print(f"Execution time: {end_time - start_time} seconds")

def mask_rebuild(args, subset, ref_subset, scalers, prune_n=0, prune_m=0):
    W_mask = {}
    W_metric = {}
    for name in subset:
        W_mask[name] = (ref_subset[name].weight.data == 0)
        if args.retain_reconstruction:
            ref_subset[name].weight.data = torch.where(W_mask[name], subset[name].weight.data, ref_subset[name].weight.data)
        else:
            ref_subset[name].weight.data = subset[name].weight.data.clone()
        W_conbine = ref_subset[name].weight.data.float()
        W_metric[name] = torch.zeros_like(W_conbine, dtype=torch.float32)
        if args.prune_metric == "WD":
            W_metric[name] = torch.abs(W_conbine) * torch.sqrt(scalers[name])
        elif args.prune_metric == "W":
            W_metric[name] = torch.abs(W_conbine)
        elif args.prune_metric == "D":
            W_metric[name] = torch.sqrt(scalers[name])


    def find_threshold(growth_metric, prune_metric):
        if growth_metric.numel() == 0 or prune_metric.numel() == 0:
            return torch.tensor(float('inf')).to(growth_metric.device), torch.tensor(float('-inf')).to(prune_metric.device)
        growth_sort_res = torch.sort(growth_metric, descending=True)[0]
        prune_sort_res = torch.sort(prune_metric, descending=False)[0]
        min_len = min(growth_sort_res.numel(), prune_sort_res.numel())
        recons_num = torch.sum((growth_sort_res[:min_len] - prune_sort_res[:min_len]) > 0)
        if args.save_data:
            if not os.path.exists(args.save_data):
                os.makedirs(args.save_data)
            save_path = os.path.join(args.save_data, f"{name}_sort_res.pth")
            if not os.path.exists(save_path):
                torch.save((growth_sort_res[:min_len] - prune_sort_res[:min_len])[:recons_num], save_path)
        if int(recons_num * args.threshold) == growth_sort_res.numel():
            growth_threshold = growth_sort_res[-1]
        else:
            growth_threshold = growth_sort_res[int(recons_num * args.threshold)]
        if int(recons_num * args.threshold) == prune_sort_res.numel():
            prune_threshold = prune_sort_res[-1]
        else:
            prune_threshold = prune_sort_res[int(recons_num * args.threshold)]
        return growth_threshold, prune_threshold
    
    if DEBUG:
        header  = "+--------------------+--------+-------+------------+-----------+----------+----------+"
        columns = "|        Name        |  Zero  |  Nan  | Growth Num | Prune Num | Sparsity | Time (s) |"
        print(header+'\n'+columns+'\n'+header)
        
    growth_threshold, prune_threshold = None, None
    
    if prune_n == 0 and args.prune_granularity == "block":
        # unstructured sparsity block
        growth_metric = torch.cat([W_metric[name][W_mask[name]] for name in subset])
        prune_metric = torch.cat([W_metric[name][~W_mask[name]] for name in subset])
        growth_threshold, prune_threshold = find_threshold(growth_metric, prune_metric)
        
    for name in subset:
        start_time = time.time()
        growth_indices, prune_indices = torch.zeros_like(W_mask[name]), torch.zeros_like(W_mask[name])
        if prune_n != 0:
            # N:M structured sparsity
            growth_threshold = torch.zeros(W_metric[name].shape[0]).to(W_metric[name].device)
            for j in range(W_metric[name].shape[0]):
                growth_metric = W_metric[name][j][W_mask[name][j]]
                prune_metric = W_metric[name][j][~W_mask[name][j]]
                growth_threshold[j], _ = find_threshold(growth_metric, prune_metric)
            growth_threshold = growth_threshold.reshape(-1, 1)
            growth_indices = (W_metric[name] > growth_threshold) & (W_mask[name])
            for ii in range(W_metric[name].shape[1]):
                if ii % prune_m == 0:
                    recons_num = torch.sum(growth_indices[:, ii:ii+prune_m], dim=1).reshape(-1, 1)
                    tmp_metric = torch.where(~W_mask[name][:, ii:ii+prune_m], W_metric[name][:, ii:ii+prune_m], torch.tensor(float('inf')).to(W_metric[name].device))
                    sort_res = torch.sort(tmp_metric, dim=1, descending=False)[0]
                    prune_threshold = torch.gather(sort_res, dim=1, index=recons_num).reshape(-1, 1)
                    prune_indices[:, ii:ii+prune_m] = (W_metric[name][:, ii:ii+prune_m] < prune_threshold) & (~W_mask[name][:, ii:ii+prune_m])
        else:
            # unstructured sparsity
            if args.prune_granularity == "layer":
                growth_metric = W_metric[name][W_mask[name]]
                prune_metric = W_metric[name][~W_mask[name]]
                growth_threshold, prune_threshold = find_threshold(growth_metric, prune_metric)
                
            elif args.prune_granularity == "output1":
                growth_threshold = torch.zeros(W_metric[name].shape[0]).to(W_metric[name].device)
                prune_threshold = torch.zeros(W_metric[name].shape[0]).to(W_metric[name].device)
                for j in range(W_metric[name].shape[0]):
                    growth_metric = W_metric[name][j][W_mask[name][j]]
                    prune_metric = W_metric[name][j][~W_mask[name][j]]
                    growth_threshold[j], prune_threshold[j] = find_threshold(growth_metric, prune_metric)
                growth_threshold = growth_threshold.reshape(-1, 1)
                prune_threshold = prune_threshold.reshape(-1, 1)
            elif args.prune_granularity == "input1":
                growth_threshold = torch.zeros(W_metric[name].shape[1]).to(W_metric[name].device)
                prune_threshold = torch.zeros(W_metric[name].shape[1]).to(W_metric[name].device)
                for j in range(W_metric[name].shape[1]):
                    growth_metric = W_metric[name][:, j][W_mask[name][:, j]]
                    prune_metric = W_metric[name][:, j][~W_mask[name][:, j]]
                    growth_threshold[j], prune_threshold[j] = find_threshold(growth_metric, prune_metric)
                growth_threshold = growth_threshold.reshape(1, -1)
                prune_threshold = prune_threshold.reshape(1, -1)

            growth_indices = W_metric[name] > growth_threshold
            prune_indices = W_metric[name] < prune_threshold

        if DEBUG:
            growth_num = torch.sum(growth_indices[W_mask[name]])
            prune_num = torch.sum(prune_indices[~W_mask[name]])
        
        W_mask[name][growth_indices] = False        
        W_mask[name][prune_indices] = True
        
        ref_subset[name].weight.data[W_mask[name]] = 0  ## set weights to zero
        sparsity = torch.sum(W_mask[name]).item() / W_mask[name].numel()
        end_time = time.time()
        excution_time = end_time - start_time
        if DEBUG:
            zero_num, nan_num = torch.sum(W_metric[name] == 0), torch.sum(torch.isnan(W_metric[name]))
            row = f"| {name:<18} | {zero_num:<6} | {nan_num:<5} | {growth_num:<10} | {prune_num:<9} | {sparsity:<8.2%} | {excution_time:<9.2f} |"
            print(row+'\n'+header)

def prune_barber(args, dense_model, sparse_model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    dense_use_cache = dense_model.config.use_cache 
    dense_model.config.use_cache = False 
    sparse_use_cache = sparse_model.config.use_cache
    sparse_model.config.use_cache = False

    config = dense_model.config

    print("loading calibdation data")
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=dense_model.seqlen,tokenizer=tokenizer)
    print("dataset loading complete")
    
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(dense_model, dataloader, args.nsamples, device)
    
    dense_layers = dense_model.model.layers
    sparse_layers = sparse_model.model.layers

    progress_bar = tqdm(range(len(sparse_layers)*args.nsamples*2), desc="Rebuilding compressed model: ")
    for i in range(len(sparse_layers)):
        if DEBUG:
            print(f"Rebuilding layer {i} ...")
        dense_layer = dense_layers[0]
        sparse_layer = sparse_layers[i]

        ref_subset = find_layers(sparse_layer)
        
        if f"model.layers.{i}" in dense_model.hf_device_map:   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = dense_model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)
            
        # self-attn
        attention = dense_layer.self_attn
        attn_subset = find_layers(attention,name="self_attn")
        attentiongpt = LlamaAttentionGPT(attention, attention_mask=attention_mask, position_ids=position_ids)
    
        def input_layernorm():
            def tmp(_, inp, out):
                attentiongpt.get_input(out[0].data)
            return tmp
    
        def self_attn():
            def tmp(_, inp, out):
                attentiongpt.get_output(out[0][0].data)
                attentiongpt.add_batch()
            return tmp

        handles = []
        handles.append(sparse_layer.input_layernorm.register_forward_hook(input_layernorm()))
        handles.append(sparse_layer.self_attn.register_forward_hook(self_attn()))
        for j in range(args.nsamples):
            temp = sparse_layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
            outs[j] = temp.clone().detach()
            del temp
            torch.cuda.empty_cache()
            progress_bar.update(1)
        for h in handles:
            h.remove()
            
        if DEBUG:
            print("Attention Pre Loss:", attentiongpt.pre_loss)
        
        scalers = {}
        scalers.update(attentiongpt.scalers)

        mask_rebuild(args, attn_subset, ref_subset, scalers, prune_n=prune_n, prune_m=prune_m)
        attentiongpt.nsamples = 0

        def self_attn():
            def tmp(_, inp, out):
                attentiongpt.update_loss(out[0][0].data)
            return tmp
        
        handles = []
        handles.append(sparse_layer.self_attn.register_forward_hook(self_attn()))
        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = sparse_layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        if DEBUG:
            print(f"Attention Post Loss:",attentiongpt.post_loss)
            
        attentiongpt.free()
        torch.cuda.empty_cache()
        
        # mlp
        mlp = dense_layer.mlp
        mlp_subset = find_layers(mlp, name="mlp")
        mlpgpt = LlamaMLPGPT(mlp)

        def mlp():
            def tmp(_, inp, out):
                mlpgpt.get_input(inp[0].data)
                mlpgpt.get_output(out[0].data)
                mlpgpt.add_batch()
            return tmp
        
        handles = []
        handles.append(sparse_layer.mlp.register_forward_hook(mlp()))
        for j in range(args.nsamples):
            temp = sparse_layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
            outs[j] = temp.clone().detach()
            del temp
            torch.cuda.empty_cache()
            progress_bar.update(1)
        for h in handles:
            h.remove()

        if DEBUG:
            print("MLP Pre Loss:", mlpgpt.pre_loss)

        scalers = {}
        scalers.update(mlpgpt.scalers)

        mask_rebuild(args, mlp_subset, ref_subset, scalers, prune_n=prune_n, prune_m=prune_m)
        mlpgpt.nsamples = 0

        def mlp():
            def tmp(_, inp, out):
                mlpgpt.update_loss(out[0].data)
            return tmp
        
        handles = []
        handles.append(sparse_layer.mlp.register_forward_hook(mlp()))
        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = sparse_layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        if DEBUG:
            print(f"MLP Post Loss:", mlpgpt.post_loss)

        mlpgpt.free()
        torch.cuda.empty_cache()
        
        del dense_layers[0]
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    dense_model.config.use_cache = dense_use_cache
    sparse_model.config.use_cache = sparse_use_cache
    torch.cuda.empty_cache()