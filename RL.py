




import argparse
import numpy as np
import torch
import gym
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from transformers import AutoModelForCausalLM, LlamaTokenizer
from datasets import load_dataset
from lib.data import get_loaders
from lib.layerwrapper import WrappedGPT
import warnings

#############################################
# 模型加载函数，每次调用返回新模型
def get_llm(model_path, cache_dir="/root/autodl-tmp/llm_weights"):
    print("Loading model from", model_path, flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        cache_dir=cache_dir,
        low_cpu_mem_usage=True,
        device_map="auto"
    )
    model.seqlen = model.config.max_position_embeddings
    print("Model loaded. seqlen =", model.seqlen, flush=True)
    return model

#############################################
# Calibration 函数：在 CPU 上创建校准数据
def prepare_calibration_input(model, dataloader, device, nsamples):
    print("Preparing calibration data on CPU...", flush=True)
    layers = model.model.layers if hasattr(model.model, "layers") else model.model.decoder.layers
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
    print("Calibration data prepared.", flush=True)
    return inps, outs, attention_mask, position_ids

def prepare_calibration_input_opt(model, dataloader, device, nsamples):
    print("Preparing calibration data for OPT model on CPU...", flush=True)
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
    print("Calibration data for OPT prepared.", flush=True)
    return inps, outs, attention_mask, None

#############################################
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

#############################################
# Wanda 剪枝函数（使用预加载的校准数据）
def prune_wanda_ww_cached(args, model, tokenizer, device, prune_ratios, dataloader):
    """
    这里采用简化版本的 Wanda：先获取校准数据，然后对每一层按照 prune_ratios 进行 unstructured 剪枝。
    """
    # 先获取校准数据（在 CPU 上创建）
    inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, "cpu", nsamples=args.nsamples)
    print("inps shape:", inps.shape, flush=True)
    if hasattr(model.model, "layers"):
        layers = model.model.layers
    else:
        layers = model.model.decoder.layers
    # 假设 prune_ratios 长度等于层数，每层内部 7 个子层
    k = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)
        # 若模型有多 GPU 分布，则取对应设备，否则用传入 device
        if hasattr(model, "hf_device_map") and f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            print(f"using wanda! layer {i} device {dev}", flush=True)
        else:
            dev = device
        for name in subset:
            print(f"Pruning layer {i} name {name}", flush=True)
            W = subset[name].weight.data
            num_to_prune = int(W.shape[1] * prune_ratios[i])
            sort_res = torch.sort(torch.abs(W), dim=-1, stable=True)
            indices = sort_res[1][:, :num_to_prune]
            W_mask = torch.zeros_like(W, dtype=torch.bool)
            W_mask.scatter_(1, indices, True)
            W[W_mask] = 0
        k += 1
    torch.cuda.empty_cache()
    print("Wanda pruning finished.", flush=True)

#############################################
# 定义一个函数，在同一进程中加载模型、运行 Wanda 剪枝，并返回 loss
def evaluate_pruning(pruning_params, args, model_path, cache_dir, tokenizer, inputs):
    print("Loading model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        cache_dir=cache_dir,
        device_map="auto",
        torch_dtype=torch.float16
    )
    model.seqlen = model.config.max_position_embeddings
    print("Model loaded.", flush=True)
    dataloader, _ = get_loaders("c4", nsamples=args.nsamples, seed=args.seed,
                                 seqlen=model.seqlen, tokenizer=tokenizer)
    print("Loading calibration data...", flush=True)
    with torch.no_grad():
        if hasattr(model.model, "layers"):
            cal_data = prepare_calibration_input(model, dataloader, "cpu", args.nsamples)
        else:
            cal_data = prepare_calibration_input_opt(model, dataloader, "cpu", args.nsamples)
    print("Calibration data loaded.", flush=True)
    prune_wanda_ww_cached(args, model, tokenizer, "cuda", prune_ratios=pruning_params, dataloader=dataloader)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss.item()
    print("Evaluation finished. Loss =", loss, flush=True)
    del model
    torch.cuda.empty_cache()
    return loss

#############################################
# 回调函数：跟踪训练进度
class ProgressCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(ProgressCallback, self).__init__(verbose)
        self.episode_count = 0
    def _on_step(self) -> bool:
        return True
    def _on_rollout_end(self) -> None:
        self.episode_count += 1
        print(f"Rollout {self.episode_count} finished at step {self.num_timesteps}.", flush=True)

#############################################
# RL 环境：每个 episode 重新加载模型、执行剪枝并计算 loss
class PruningEnv(gym.Env):
    """
    RL 环境：
      - 每次 step() 重新加载模型、加载数据、执行 Wanda 剪枝、计算剪枝后 loss。
      - RL 动作为每层的 ESD 权重（用于融合 ESD 与 importance）。
    """
    def __init__(self, model_path, esd_ratios, importance_scores, args, cache_dir):
        super(PruningEnv, self).__init__()
        self.model_path = model_path
        self.cache_dir = cache_dir
        self.args = args
        self.esd_ratios = esd_ratios
        self.importance_scores = importance_scores
        self.num_layers = len(esd_ratios)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.action_space = gym.spaces.Box(low=0.0, high=1.0,
                                           shape=(self.num_layers,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0,
                                                shape=(self.num_layers*2,), dtype=np.float32)
        # 预先加载数据集，用于 loss 评估
        dataset = load_dataset("roneneldan/TinyStories", split="train")
        sample_texts = [dataset[i]["text"] for i in range(100)]
        tokenizer_name = "HuggingFaceM4/llama-7b-tokenizer"
        tokenizer = LlamaTokenizer.from_pretrained(tokenizer_name)
        inputs = tokenizer(sample_texts, return_tensors="pt", padding=True,
                           truncation=True, max_length=256)
        self.inputs = {k: v for k, v in inputs.items()}  # 注意：后面会移至模型 device

    def reset(self):
        return np.concatenate([self.esd_ratios, self.importance_scores])

    def step(self, action):
        # 根据动作计算每层剪枝比例
        esd_weights = np.clip(action, 0.0, 1.0)
        final_pruning_ratios = esd_weights * self.esd_ratios + (1 - esd_weights) * self.importance_scores

        # 重新加载模型、执行 Wanda 剪枝并计算 loss
        loss = evaluate_pruning(final_pruning_ratios, self.args, self.model_path, self.cache_dir, LlamaTokenizer.from_pretrained(self.model_path), self.inputs)
        # 计算 reward，loss 越低 reward 越高（负向 reward）
        reward = -loss
        done = True
        obs = np.concatenate([self.esd_ratios, self.importance_scores])
        return obs, reward, done, {}

#############################################
# 主程序
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="pinkmanlove/llama-7b-hf")
    parser.add_argument('--cache_dir', type=str, default="/root/autodl-tmp/llm_weights")
    parser.add_argument('--sparsity_ratio', type=float, default=0.7)
    parser.add_argument('--nsamples', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use_variant', action='store_true')
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model_path = args.model
    cache_dir = args.cache_dir

    # baseline：加载一次模型计算剪枝前 loss（可选）
    model = get_llm(model_path, cache_dir)
    tokenizer_name = "HuggingFaceM4/llama-7b-tokenizer"
    tokenizer = LlamaTokenizer.from_pretrained(tokenizer_name)
 

    dataset = load_dataset("roneneldan/TinyStories", split="train")
    sample_texts = [dataset[i]["text"] for i in range(100)]
    inputs = tokenizer(sample_texts, return_tensors="pt", padding=True,
                       truncation=True, max_length=256)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        base_loss = outputs.loss.item()
    print(f"🚀 剪枝前 Loss: {base_loss:.6f}", flush=True)
    del model
    torch.cuda.empty_cache()

    # 这里 esd_ratios 和 importance_scores 可由外部预先计算好，此处示例使用随机数
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

    env = PruningEnv(model_path, esd_ratios, importance_scores, args, cache_dir)

    callback = ProgressCallback(verbose=1)
    model_rl = PPO("MlpPolicy", env, verbose=1, device='cpu')
    model_rl.learn(total_timesteps=5000, callback=callback)

    best_action = model_rl.predict(env.reset())[0]
    print(f"🚀 最优 ESD 权重（每层）: {best_action}", flush=True)
    final_pruning_ratios = best_action * esd_ratios + (1 - best_action) * importance_scores
    print("🔥 RL 计算的最终剪枝比例:", final_pruning_ratios, flush=True)
    np.save("final_pruning_ratios_rl.npy", final_pruning_ratios)
