import argparse
import copy
import gym
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from transformers import AutoModelForCausalLM, LlamaTokenizer
from datasets import load_dataset
from lib.layerwrapper import WrappedGPT
from lib.data import get_loaders
import warnings

#############################################
# Calibration 函数：在 CPU 上创建校准数据
def prepare_calibration_input(model, dataloader, device, nsamples):
    layers = model.model.layers
    init_device = "cpu"
    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((nsamples, model.seqlen, model.config.hidden_size),
                       dtype=dtype, device=init_device)
    inps.requires_grad = False
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp.cpu()
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs.get('position_ids', None)
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(init_device))
        except ValueError:
            pass 
    layers[0] = layers[0].module
    torch.cuda.empty_cache()
    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    return inps, outs, attention_mask, position_ids

def prepare_calibration_input_opt(model, dataloader, device, nsamples):
    layers = model.model.decoder.layers
    init_device = "cpu"
    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((nsamples, model.seqlen, model.config.hidden_size),
                       dtype=dtype, device=init_device)
    inps.requires_grad = False
    cache = {'i': 0, 'attention_mask': None}
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp.cpu()
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(init_device))
        except ValueError:
            pass 
    layers[0] = layers[0].module
    torch.cuda.empty_cache()
    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    return inps, outs, attention_mask, None

#############################################
# 工具函数
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

#############################################
# Wanda 剪枝函数（使用预加载的校准数据）
def prune_wanda_ww_cached(args, model, tokenizer, device, prune_ratios, cal_data):
    s1 = 1.0 - args.epsilon
    s2 = 1.0 + args.epsilon
    res = []
    for j in prune_ratios:
        for i in range(7):
            res.append(j)
    res = np.array(res)
    inps, outs, attention_mask, position_ids = cal_data
    print("inps", inps.shape)
    if "OPT" in model.__class__.__name__:
        layers = model.model.decoder.layers
    else:
        layers = model.model.layers
    layer_num = len(find_layers(layers))
    if res is None:
        res = np.array([args.sparsity_ratio for _ in range(layer_num * 7)])
    
    k = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)
        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            print(f"using wanda! layer {i} device {dev}")
        else:
            dev = device
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
                sample_inp = inps[j].unsqueeze(0).to(dev)
                sample_attention = attention_mask.to(dev)
                if position_ids is not None:
                    sample_position_ids = position_ids.to(dev)
                else:
                    sample_position_ids = None
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(sample_inp, attention_mask=sample_attention)[0]
                else:
                    outs[j] = layer(sample_inp, attention_mask=sample_attention, position_ids=sample_position_ids)[0]
        for h in handles:
            h.remove()
        for name in subset:
            print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1)))
            W_mask = (torch.zeros_like(W_metric) == 1)
            if args.use_variant:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)
                tmp_metric = torch.cumsum(sort_res[0], dim=1)
                sum_before = W_metric.sum(dim=1)
                alpha = 0.4
                alpha_hist = [0., 0.8]
                W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                while (torch.abs(cur_sparsity - args.sparsity_ratio) > 0.001) and (alpha_hist[1]-alpha_hist[0] >= 0.001):
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
                sort_res = torch.sort(W_metric, dim=-1, stable=True)
                indices = sort_res[1][:, :int(W_metric.shape[1]*res[k])]
                k += 1
                W_mask.scatter_(1, indices, True)
            subset[name].weight.data[W_mask] = 0
        for j in range(args.nsamples):
            with torch.no_grad():
                sample_inp = inps[j].unsqueeze(0).to(dev)
                sample_attention = attention_mask.to(dev)
                if position_ids is not None:
                    sample_position_ids = position_ids.to(dev)
                else:
                    sample_position_ids = None
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(sample_inp, attention_mask=sample_attention)[0]
                else:
                    outs[j] = layer(sample_inp, attention_mask=sample_attention, position_ids=sample_position_ids)[0]
        inps = outs.cpu()
    model.config.use_cache = use_cache
    torch.cuda.empty_cache()

#############################################
# RL 环境：每个 episode 重新加载模型，并预先加载校准数据，RL 仅调整参数
class PruningEnv(gym.Env):
    """
    RL 环境：
      - 每个 episode 重新加载模型（通过 model_loader）。
      - 在 reset() 中预先加载校准数据并缓存到 self.cal_data。
      - RL 动作为每层的 ESD 权重（用于融合 ESD 和 GradNorm）。
      - 在 step() 中调用 prune_wanda_ww_cached()，然后计算剪枝后模型 loss 作为 reward。
    """
    def __init__(self, model_loader, esd_ratios, importance_scores, args, tokenizer, device, inputs, base_loss):
        super(PruningEnv, self).__init__()
        self.model_loader = model_loader  # 每次返回新模型
        self.esd_ratios = esd_ratios
        self.importance_scores = importance_scores
        self.num_layers = len(esd_ratios)
        self.args = args
        self.tokenizer = tokenizer
        self.device = device
        self.inputs = inputs
        self.base_loss = base_loss
        self.action_space = gym.spaces.Box(low=0.0, high=1.0, shape=(self.num_layers,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(self.num_layers*2,), dtype=np.float32)
        self.cal_data = None

    def reset(self):
        self.model = self.model_loader()
        if not hasattr(self.model, 'seqlen'):
            self.model.seqlen = self.model.config.max_position_embeddings
        dataloader, _ = get_loaders("c4", nsamples=self.args.nsamples,
                                    seed=self.args.seed, seqlen=self.model.seqlen,
                                    tokenizer=self.tokenizer)
        with torch.no_grad():
            if "OPT" in self.model.__class__.__name__:
                self.cal_data = prepare_calibration_input_opt(self.model, dataloader, self.device, self.args.nsamples)
            else:
                self.cal_data = prepare_calibration_input(self.model, dataloader, self.device, self.args.nsamples)
        return np.concatenate([self.esd_ratios, self.importance_scores])
    
    def step(self, action):
        esd_weights = np.clip(action, 0.0, 1.0)
        final_pruning_ratios = esd_weights * self.esd_ratios + (1 - esd_weights) * self.importance_scores
        prune_wanda_ww_cached(self.args, self.model, self.tokenizer, self.device,
                                prune_ratios=final_pruning_ratios, cal_data=self.cal_data)
        with torch.no_grad():
            outputs = self.model(**self.inputs, labels=self.inputs["input_ids"])
            pruned_loss = outputs.loss.item()
        loss_increase = (pruned_loss - self.base_loss) / self.base_loss
        reward = -loss_increase
        done = True
        obs = np.concatenate([self.esd_ratios, self.importance_scores])
        del self.model
        del self.cal_data
        torch.cuda.empty_cache()
        return obs, reward, done, {}

#############################################
# 主程序
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="pinkmanlove/llama-7b-hf", help="LLaMA 模型路径")
    parser.add_argument('--cache_dir', type=str, default="/root/autodl-tmp/llm_weights", help="模型缓存路径")
    parser.add_argument('--ww_metric', type=str, default="alpha_peak", help="ESD 剪枝指标")
    parser.add_argument('--ww_metric_cache', type=str, default="./data/llama-7b-hf", help="ESD 指标存储路径")
    parser.add_argument('--sparsity_ratio', type=float, default=0.7, help="目标剪枝比例")
    parser.add_argument('--prune_method', type=str, default="wanda_ww", help="剪枝方法")
    parser.add_argument('--epsilon', type=float, default=0.2, help="剪枝比例的微调范围")
    parser.add_argument('--nsamples', type=int, default=10, help="校准样本数")
    parser.add_argument('--seed', type=int, default=42, help="随机种子")
    parser.add_argument('--use_variant', action='store_true', help="是否使用 Wanda variant 剪枝")
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 定义模型加载函数，每次返回新模型
    model_loader = lambda: AutoModelForCausalLM.from_pretrained(
        args.model, cache_dir=args.cache_dir, device_map="auto", torch_dtype=torch.float16
    )
    
    # 预先加载一次模型用于 baseline 评估
    model = model_loader()
    tokenizer_name = "HuggingFaceM4/llama-7b-tokenizer"
    tokenizer = LlamaTokenizer.from_pretrained(tokenizer_name)
    if not hasattr(model, 'seqlen'):
        model.seqlen = model.config.max_position_embeddings
    
    dataset = load_dataset("roneneldan/TinyStories", split="train")
    sample_texts = [dataset[i]["text"] for i in range(100)]
    inputs = tokenizer(sample_texts, return_tensors="pt", padding=True, truncation=True, max_length=256)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        base_loss = outputs.loss.item()
    print(f"🚀 剪枝前 LLaMA-7B 在 TinyStories Loss: {base_loss:.6f}")
    
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
    
    # 创建 RL 环境：每个 episode 重新加载模型和预先加载校准数据
    env = PruningEnv(model_loader, esd_ratios, importance_scores, args, tokenizer, device, inputs, base_loss)
    model_rl = PPO("MlpPolicy", env, verbose=1, device='cpu')  # 用 CPU 训练策略网络
    model_rl.learn(total_timesteps=5000)
    
    best_action = model_rl.predict(env.reset())[0]
    print(f"🚀 最优 ESD 权重（每层）: {best_action}")
    final_pruning_ratios = best_action * esd_ratios + (1 - best_action) * importance_scores
    print("🔥 RL 计算的最终剪枝比例:", final_pruning_ratios)
    np.save("final_pruning_ratios_rl.npy", final_pruning_ratios)
