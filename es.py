import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from transformers import LlamaTokenizer, AutoModelForCausalLM
from scipy.stats import norm
from lib.data import get_loaders
from lib.layerwrapper import WrappedGPT
from tqdm import tqdm
import random
# ========== 1. 获取 LLM 模型 ==========
# 工具函数：递归查找指定类型的层
def find_layers(module, layers=[nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(child, layers=layers,
                               name=name + '.' + name1 if name != '' else name1))
    return res

def return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before):
    thres_cumsum = sum_before * alpha 
    sort_mask = tmp_metric <= thres_cumsum.reshape((-1,1))
    thres = torch.gather(sort_res[0], dim=1,
                         index=sort_mask.sum(dim=1, keepdims=True)-1)
    W_mask = (W_metric <= thres)
    cur_sparsity = (W_mask==True).sum() / W_mask.numel()
    return W_mask, cur_sparsity

def get_llm(model_path, cache_dir="llm_weights"):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        cache_dir=cache_dir,
        low_cpu_mem_usage=True,
        device_map="auto"
    )
    model.seqlen = 2048
    return model

def prune_wanda(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0, ratios=None):
    # 确保计算的数据都在 model.device
    device = model.device
    
    use_cache = model.config.use_cache 
    model.config.use_cache = False 
    res = []
    for i in ratios:
        for j in range(7):
            res.append(i)
    ratios = np.array([i.cpu().numpy() if isinstance(i, torch.Tensor) else i for i in res])


    #print(ratios)
    #print("loading calibdation data")
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
    #print("dataset loading complete")
    with torch.no_grad():
        if "OPT" in model.__class__.__name__:
            inps, outs, attention_mask, position_ids = prepare_calibration_input_opt(model, dataloader, device)
        else:
            inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)


    #print ("inps",inps)

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
            #print("using wanda!")
            dev = model.hf_device_map[f"model.layers.{i}"]
            #print(f"layer {i} device {dev}")
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
            #print(f"pruning layer {i} name {name}")
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
def calculate_sparsity(model):
    total_params = 0
    zero_params = 0
    for name, param in model.named_parameters():
        if "weight" in name:  # 只计算权重，不计算 bias
            total_params += param.numel()
            zero_params += (param == 0).sum().item()
    sparsity = zero_params / total_params
    print(f"📉 Model Sparsity: {sparsity:.4f} ({zero_params}/{total_params} zero values)")
    return sparsity
# ========== 2. 定义剪枝环境 ==========
class LayerPruningOptimization:
    def __init__(self, model_path, cache_dir, dataset, tokenizer, esd_ratios, importance_scores, args):
        self.model_path = model_path
        self.cache_dir = cache_dir
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.num_layers = len(esd_ratios)
        self.esd_ratios = esd_ratios
        self.importance_scores = importance_scores
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def evaluate_loss(self, weights):
        """计算当前 `weights` (混合比例) 下剪枝后模型的 loss"""

        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        weights_np = weights.cpu().numpy() if isinstance(weights, torch.Tensor) else weights  # 确保是 NumPy 数组

        esd_contrib = self.esd_ratios * weights_np
        imp_contrib = self.importance_scores * (1 - weights_np)
        layer_weights = esd_contrib + imp_contrib  # 计算最终混合权重

        # ✅ 确保 layer_weights 仍然是 NumPy 数组
        layer_weights = layer_weights.astype(np.float32)

        # 加载 LLM 模型
        model = get_llm(self.model_path, self.cache_dir)
      
    

        try:
   
            prune_wanda(self.args, model, self.tokenizer, self.device, ratios=layer_weights)
           
            

            # 评估剪枝后 loss
            sample_texts = random.sample(self.dataset, 100)
            #sample_texts = [self.dataset[i]["text"] for i in range(100)]
            inputs = self.tokenizer(sample_texts, return_tensors="pt", padding=True, truncation=True, max_length=256)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss.item()
                print(f"📉 Generation Loss History: {loss:.6f}")

        except Exception as e:
            print(f"❌ Evaluation failed: {e}")
            loss = float("inf")  # 避免异常导致 ES 失败
        finally:
            # 释放模型
            del model
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        return loss, layer_weights





# ========== 3. 进化策略 (ES) ==========
from tqdm import tqdm
import sys

class EvolutionStrategy:
    def __init__(self, env, population_size=20, sigma=0.1, alpha=0.02, generations=50):
        self.env = env
        self.population_size = population_size
        self.sigma = sigma
        self.alpha = alpha
        self.generations = generations
        self.num_layers = env.num_layers
        
        # ✅ 解决方案：添加 self.device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def optimize(self):
        """ 运行进化策略进行优化 """
        weights_np = 0.8 * self.env.esd_ratios + 0.2 * self.env.importance_scores  # 计算初始权重
        best_loss = float("inf")
        best_weights = weights_np

        print("🚀 开始 RL 训练", flush=True)
        progress_bar = tqdm(range(self.generations), desc="Training Progress", file=sys.stdout, ascii=True)

        for gen in progress_bar:
            noise = np.random.randn(self.population_size, self.num_layers)  # 生成噪声

            population = weights_np + self.sigma * noise  # 生成新种群

            rewards = np.zeros(self.population_size)
            for i in range(self.population_size):
                loss, _ = self.env.evaluate_loss(population[i])  # 计算当前个体的 loss
                rewards[i] = -loss  # 目标是最小化 loss
                torch.cuda.empty_cache()

            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
            gradient = np.dot(noise.T, rewards) / self.population_size  # 计算梯度
            print("gradient", gradient)
            print("reward: ", rewards)
            # ✅ 确保 gradient 是 NumPy 数组
            gradient = gradient.cpu().numpy() if isinstance(gradient, torch.Tensor) else gradient

            # ✅ 进行 weights 更新
            weights_np += self.alpha * gradient

            # ✅ 重新转换为 PyTorch Tensor，放回 CUDA
            weights = torch.tensor(weights_np, dtype=torch.float32, device=self.device)

            # 计算当前 generation 的最终 loss
            loss, final_weights = self.env.evaluate_loss(weights)

            # ✅ 直接打印 Loss 和参数（限制小数点后 4 位）
            print(f"\n🌀 Generation {gen+1}/{self.generations} | Loss: {loss:.6f}")
            print("📌 Layer Weights:", np.round(weights.cpu().numpy(), 4))  # 限制 4 位小数
            print("-" * 60)  # 让日志更清晰

            if loss < best_loss:
                best_loss = loss
                best_weights = final_weights

            progress_bar.set_postfix({"Best Loss": f"{best_loss:.6f}"})
            progress_bar.refresh()

        return best_weights, best_loss



# ========== 4. 运行优化 ==========
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="pinkmanlove/llama-7b-hf")
    parser.add_argument('--cache_dir', type=str, default="/root/autodl-tmp/llm_weights")
    parser.add_argument('--sparsity_ratio', type=float, default=0.7)
    parser.add_argument('--nsamples', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use_variant', action='store_true')
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model_path = args.model
    cache_dir = args.cache_dir

    # 加载 tokenizer
    tokenizer_name = "HuggingFaceM4/llama-7b-tokenizer"
    tokenizer = LlamaTokenizer.from_pretrained(tokenizer_name)

    # 加载 TinyStories 数据集
    dataset = load_dataset("roneneldan/TinyStories", split="train")

    # 预设的 ESD Ratios 和 Importance Scores
    num_layers = 32
    esd_ratios = np.array([
        0.57042164, 0.61759788, 0.63153112, 0.63073802, 0.65285629, 0.6482451,
        0.63005912, 0.5921672,  0.59738964, 0.56803465, 0.58708227, 0.58937198,
        0.59899241, 0.61086321, 0.61877495, 0.66812801, 0.65868002, 0.71560568,
        0.79057246, 0.74378908, 0.79461485, 0.82483709, 0.77005184, 0.76292461,
        0.81216604, 0.85205203, 0.8312614,  0.84147072, 0.78692085, 0.82967305,
        0.84142309, 0.73170304
    ])
    importance_scores = np.array([
        0, 0.25823024, 0.64031047, 0.672687, 0.7274453, 0.7274453,
        0.7274453, 0.7274453, 0.7274453, 0.7274453, 0.7274453, 0.7274453,
        0.7462126, 0.7462126, 0.7462126, 0.74832606, 0.74832606, 0.74832606,
        0.74936193, 0.74936193, 0.74936193, 0.74992037, 0.74992037, 0.74992037,
        0.75013256, 0.75013256, 0.75013256, 0.75013256, 0.75394976, 0.75394976,
        0.7545204,  0.764797
    ])

    # 初始化环境
    env = LayerPruningOptimization(model_path, cache_dir, dataset, tokenizer, esd_ratios, importance_scores, args)
    print("env done")
    # 运行进化策略优化
    es = EvolutionStrategy(env, population_size=2, sigma=0.3, alpha=0.07, generations=5)
    
    best_weights, best_loss = es.optimize()

    print("\n🔍 最优混合比例：", best_weights)
    print(f"✅ 剪枝后最优 Loss: {best_loss:.6f}")
