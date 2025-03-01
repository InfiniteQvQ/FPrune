import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

class WrappedGPT:
    """收集激活统计量"""
    def __init__(self, layer):
        self.layer = layer
        self.activations = None
        self.scaler_row = None
    
    def add_batch(self, inp, out):
        # 计算输入特征的L2范数
        inp = inp.detach()
        if self.scaler_row is None:
            self.scaler_row = torch.zeros(inp.shape[-1], device=inp.device)
        self.scaler_row += inp.pow(2).sum(dim=0)

class GQAWandaPruner:
    def __init__(self, model, tokenizer, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.config = model.config
        
        # 获取GQA配置
        self.num_attention_heads = self.config.num_attention_heads
        self.num_key_value_heads = getattr(self.config, "num_key_value_heads", self.num_attention_heads)
        self.head_dim = self.config.hidden_size // self.num_attention_heads
        self.groups_per_layer = self.num_attention_heads // self.num_key_value_heads

    def prepare_calibration_input(self, nsamples=128, seqlen=4096):
        """准备校准输入（LLaMA3专用）"""
        dummy_input = torch.randint(0, self.tokenizer.vocab_size, (nsamples, seqlen)).to(self.device)
        layers = self.model.model.layers
        dtype = next(self.model.parameters()).dtype
        
        # 初始化缓存
        inps = torch.zeros((nsamples, seqlen, self.config.hidden_size), dtype=dtype, device=self.device)
        cache = {'i': 0}

        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module
            def forward(self, inp, **kwargs):
                inps[cache['i']] = inp
                cache['i'] += 1
                cache['attention_mask'] = kwargs.get('attention_mask')
                cache['position_ids'] = kwargs.get('position_ids')
                raise ValueError

        # 捕获输入
        layers[0] = Catcher(layers[0])
        try:
            self.model(dummy_input)
        except ValueError:
            pass
        layers[0] = layers[0].module  # 恢复原始层
        
        return inps, cache

    def prune(self, sparsity_ratio=0.5, per_group_ratios=None, nsamples=128):
        """执行GQA剪枝"""
        self.model.eval()
        original_use_cache = self.config.use_cache 
        self.config.use_cache = False

        # 准备校准数据
        print("Preparing calibration inputs...")
        inps, cache = self.prepare_calibration_input(nsamples)
        attention_mask = cache['attention_mask']
        position_ids = cache['position_ids']
        outs = torch.zeros_like(inps)

        # 逐层处理
        layers = self.model.model.layers
        for layer_idx in tqdm(range(len(layers)), desc="Pruning layers"):
            layer = layers[layer_idx]
            subset = find_layers(layer)

            # GQA剪枝核心
            if "self_attn" in subset:
                attn_layer = subset["self_attn"]
                q_proj = attn_layer.q_proj
                k_proj = attn_layer.k_proj
                v_proj = attn_layer.v_proj

                # 收集激活统计量
                wrapped_q = WrappedGPT(q_proj)
                wrapped_k = WrappedGPT(k_proj)
                wrapped_v = WrappedGPT(v_proj)
                
                handles = [
                    q_proj.register_forward_hook(lambda m, inp, out: wrapped_q.add_batch(inp[0], out)),
                    k_proj.register_forward_hook(lambda m, inp, out: wrapped_k.add_batch(inp[0], out)),
                    v_proj.register_forward_hook(lambda m, inp, out: wrapped_v.add_batch(inp[0], out))
                ]

                # 前向传播收集数据
                for j in range(nsamples):
                    with torch.no_grad():
                        layer(inps[j].unsqueeze(0), 
                             attention_mask=attention_mask,
                             position_ids=position_ids)
                
                # 移除hook
                for h in handles:
                    h.remove()

                # 计算组重要性（Wanda指标）
                group_imp = []
                for group_id in range(self.num_key_value_heads):
                    # 计算参数范围
                    q_start = group_id * self.groups_per_layer * self.head_dim
                    q_end = (group_id+1) * self.groups_per_layer * self.head_dim
                    kv_start = group_id * self.head_dim
                    kv_end = (group_id+1) * self.head_dim
                    
                    # Wanda重要性：|W| * sqrt(激活统计量)
                    q_imp = (torch.abs(q_proj.weight[q_start:q_end]) * 
                            torch.sqrt(wrapped_q.scaler_row)).sum()
                    k_imp = (torch.abs(k_proj.weight[kv_start:kv_end]) * 
                            torch.sqrt(wrapped_k.scaler_row)).sum()
                    v_imp = (torch.abs(v_proj.weight[kv_start:kv_end]) * 
                            torch.sqrt(wrapped_v.scaler_row)).sum()
                    
                    # 使用自定义比例或默认比例
                    if per_group_ratios is not None:
                        ratio = per_group_ratios[layer_idx][group_id]
                    else:
                        ratio = 1.0  # 默认全比例参与排序
                        
                    group_imp.append(ratio * (0.5*q_imp + 0.3*k_imp + 0.2*v_imp))

                # 选择要剪枝的组
                total_groups = len(group_imp)
                prune_num = int(total_groups * sparsity_ratio)
                _, prune_indices = torch.topk(-torch.tensor(group_imp), prune_num)

                # 生成剪枝掩码
                def create_mask(dim, indices, is_query=False):
                    mask = torch.ones(dim, dtype=torch.bool, device=self.device)
                    for idx in indices:
                        block = self.groups_per_layer*self.head_dim if is_query else self.head_dim
                        start = idx * block
                        end = start + block
                        mask[start:end] = False
                    return mask

                # 应用剪枝
                q_mask = create_mask(q_proj.out_features, prune_indices, is_query=True)
                kv_mask = create_mask(k_proj.out_features, prune_indices)
                
                q_proj.weight.data[q_mask] = 0
                k_proj.weight.data[kv_mask] = 0
                v_proj.weight.data[kv_mask] = 0

            # 更新中间表示
            with torch.no_grad():
                for j in range(nsamples):
                    outs[j] = layer(inps[j].unsqueeze(0),
                                  attention_mask=attention_mask,
                                  position_ids=position_ids)[0]
                inps, outs = outs, inps

        # 恢复配置
        self.config.use_cache = original_use_cache
        torch.cuda.empty_cache()
        return self.model

def find_layers(module, layers=[nn.Linear], name=''):
    """递归查找指定类型的层"""
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, 
            name=name + '.' + name1 if name != '' else name1
        ))
    return res

# 使用示例 ---------------------------------------------------
if __name__ == "__main__":
    # 加载模型
    cache_dir = "/root/autodl-tmp/llm_weights"
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Meta-Llama-3-8B",
        cache_dir=cache_dir,
        device_map="auto",
        torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
    
    # 初始化剪枝器
    pruner = GQAWandaPruner(model, tokenizer)
    
    # 自定义每个组的剪枝比例（示例：32层 x 8组）
    per_group_ratios = [
        [0.8 if i < 4 else 0.2 for _ in range(8)]  # 前4组剪枝率高
        for i in range(32)
    ]
    
    # 执行剪枝（50%全局稀疏度）
    pruned_model = pruner.prune(
        sparsity_ratio=0.7,
        per_group_ratios=per_group_ratios,
        nsamples=128
    )
    
    # 保存模型
    pruned_model.save_pretrained("./pruned-llama3-8b")
    tokenizer.save_pretrained("./pruned-llama3-8b")