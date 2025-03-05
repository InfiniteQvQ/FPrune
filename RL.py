import argparse
import copy
import gym
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from transformers import AutoModelForCausalLM, LlamaTokenizer  # 或 AutoTokenizer
from datasets import load_dataset
from lib.layerwrapper import WrappedGPT
from lib.data import get_loaders
import warnings

# ------------------- Calibration 函数 -------------------
def prepare_calibration_input(model, dataloader, device, nsamples):
    """
    在 CPU 上创建校准数据，然后后续按需转移到目标设备。
    """
    layers = model.model.layers
    init_device = "cpu"  # 数据先创建在 CPU 上
    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=init_device)
    inps.requires_grad = False
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp.cpu()  # 存到 CPU
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
    """
    针对 OPT 模型的校准函数，同样在 CPU 上创建数据
    """
    layers = model.model.decoder.layers
    init_device = "cpu"
    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=init_device)
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

# ------------------- 工具函数 -------------------
def find_layers(module, layers=[nn.Linear], name=''):
    """
    递归查找模块中指定类型的层
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(child, layers=layers, name=name + '.' + name1 if name != '' else name1))
    return res

def return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before):
    thres_cumsum = sum_before * alpha 
    sort_mask = tmp_metric <= thres_cumsum.reshape((-1,1))
    thres = torch.gather(sort_res[0], dim=1, index=sort_mask.sum(dim=1, keepdims=True)-1)
    W_mask = (W_metric <= thres)
    cur_sparsity = (W_mask==True).sum() / W_mask.numel()
    return W_mask, cur_sparsity

# ------------------- 剪枝函数（Wanda） -------------------
def prune_wanda_ww(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0, prune_ratios=0):
    """
    根据传入的每层剪枝比例，调用 wanda 剪枝
    """
    s1 = 1.0 - args.epsilon
    s2 = 1.0 + args.epsilon
    res = []
    for j in prune_ratios:
        for i in range(7):  # 假设每个 transformer 层内部有 7 个子层
            res.append(j)
    res = np.array(res)
    prune_wanda(args, model, tokenizer, device, ratios=res) 

def prune_wanda(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0, ratios=None):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    print("loading calibration data")
    dataloader, _ = get_loaders("c4", nsamples=args.nsamples, seed=args.seed, seqlen=model.seqlen, tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        if "OPT" in model.__class__.__name__:
            inps, outs, attention_mask, position_ids = prepare_calibration_input_opt(model, dataloader, device, args.nsamples)
        else:
            inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device, args.nsamples)
    print("inps", inps.shape)
    if "OPT" in model.__class__.__name__:
        layers = model.model.decoder.layers
    else:    
        layers = model.model.layers
    layer_num = len(find_layers(layers))
    if ratios is None:
        ratios = [args.sparsity_ratio for _ in range(layer_num)]
    
    k = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)
        # 如果该层有多 GPU 分配，则取当前层对应设备（不一次性转移整个 inps）
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
        # 对每个样本单独转移到当前层设备进行前向传播
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
            if prune_n != 0:
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:, ii:(ii+prune_m)].float()
                        W_mask.scatter_(1, ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)
                if args.use_variant:
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
                    indices = sort_res[1][:, :int(W_metric.shape[1]*ratios[k])]
                    k += 1
                    W_mask.scatter_(1, indices, True)
            subset[name].weight.data[W_mask] = 0
        # 每个样本前向计算，单独转移到当前设备
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
        # 将本轮前向结果移回 CPU，供下一层使用
        inps = outs.cpu()
    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()

# ------------------- RL 环境 -------------------
class PruningEnv(gym.Env):
    """
    RL 环境：
      - 每个 episode 从原始模型状态开始（通过 load_state_dict 恢复）。
      - 动作为每层的剪枝比例权重（ESD 部分与 GradNorm 部分组合）。
      - 环境根据动作对模型进行剪枝，然后在验证集上计算 perplexity（或 loss）作为评价，返回 reward。
    """
    def __init__(self, model, esd_ratios, importance_scores, args, tokenizer, device, inputs, base_loss):
        super(PruningEnv, self).__init__()
        self.model = model
        self.esd_ratios = esd_ratios
        self.importance_scores = importance_scores
        self.num_layers = len(esd_ratios)
        self.args = args
        self.tokenizer = tokenizer
        self.device = device
        self.inputs = inputs
        self.base_loss = base_loss
        # 保存原始模型状态（在 CPU 上保存，保证每个 episode 从干净状态开始）
        self.initial_state = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
        # 动作空间：每层的 ESD 权重，取值范围 [0,1]
        self.action_space = gym.spaces.Box(low=0.0, high=1.0, shape=(self.num_layers,), dtype=np.float32)
        # 观察空间：简单返回两组剪枝比例（ESD 与 GradNorm），这里固定不变，可根据需要扩展
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(self.num_layers*2,), dtype=np.float32)

    def reset(self):
        # 每个 episode 开始时，恢复原始模型状态
        self.model.load_state_dict(self.initial_state)
        self.esd_weights = np.ones(self.num_layers) * 0.8  # 初始默认动作
        return np.concatenate([self.esd_ratios, self.importance_scores])

    def step(self, action):
        # 限定动作在 [0,1] 内
        self.esd_weights = np.clip(action, 0.0, 1.0)
        # 计算最终剪枝比例（这里假设最终剪枝比例由 ESD 部分和 GradNorm 部分线性组合得到）
        final_pruning_ratios = self.esd_weights * self.esd_ratios + (1 - self.esd_weights) * self.importance_scores
        # 对模型进行剪枝（调用 wanda 剪枝）
        prune_wanda_ww(self.args, self.model, self.tokenizer, self.device, prune_ratios=final_pruning_ratios)
        # 评估剪枝后的模型（计算 loss 或 perplexity）
        with torch.no_grad():
            outputs = self.model(**self.inputs, labels=self.inputs["input_ids"])
            pruned_loss = outputs.loss.item()
        # 奖励设计：以剪枝后 loss 相对于原始 loss 的变化作为奖励（loss 越低，reward 越高）
        loss_increase = (pruned_loss - self.base_loss) / self.base_loss
        reward = -loss_increase
        done = True  # 每个 episode 只有一次剪枝评估
        obs = np.concatenate([self.esd_ratios, self.importance_scores])
        return obs, reward, done, {}

# ------------------- 主程序 -------------------
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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 加载模型（使用 device_map="auto" 自动分配各层到多个 GPU）
    model = AutoModelForCausalLM.from_pretrained(args.model, cache_dir=args.cache_dir, device_map="auto", torch_dtype=torch.float16)
    tokenizer_name = "HuggingFaceM4/llama-7b-tokenizer"
    tokenizer = LlamaTokenizer.from_pretrained(tokenizer_name)

    # 如果模型没有 seqlen 属性，则使用配置中的 max_position_embeddings
    if not hasattr(model, 'seqlen'):
        model.seqlen = model.config.max_position_embeddings

    # 加载数据集，并对部分文本进行 tokenization
    dataset = load_dataset("roneneldan/TinyStories", split="train")
    sample_texts = [dataset[i]["text"] for i in range(100)]
    inputs = tokenizer(sample_texts, return_tensors="pt", padding=True, truncation=True, max_length=256)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # 计算原始模型的 loss（作为 baseline），这里 loss 越低表示模型效果越好
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        base_loss = outputs.loss.item()
    print(f"🚀 剪枝前 LLaMA-7B 在 TinyStories Loss: {base_loss:.6f}")

    # 示例的 ESD 剪枝比例与 GradNorm（或其它重要性指标），这两个数组应当预先计算好
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

    # 创建 RL 环境，每个 episode 都将重新加载原始模型状态
    env = PruningEnv(model, esd_ratios, importance_scores, args, tokenizer, device, inputs, base_loss)
    # PPO 使用 MlpPolicy，此处使用默认参数，后续可根据需要调整
    model_rl = PPO("MlpPolicy", env, verbose=1)
    model_rl.learn(total_timesteps=5000)

    # 训练结束后，利用 RL 代理得到最优剪枝比例（每层的 ESD 权重）
    best_action = model_rl.predict(env.reset())[0]
    print(f"🚀 最优 ESD 权重（每层）: {best_action}")
    final_pruning_ratios = best_action * esd_ratios + (1 - best_action) * importance_scores
    print("🔥 RL 计算的最终剪枝比例:", final_pruning_ratios)
    np.save("final_pruning_ratios_rl.npy", final_pruning_ratios)
