import torch
import numpy as np
from tqdm import tqdm
from torch import nn
from transformers import LlamaForCausalLM, AutoTokenizer, LlamaTokenizer, AutoModelForCausalLM
from datasets import load_dataset

# 配置参数
class Args:
    def __init__(self):
        self.dataset = "wikitext2"
        self.seqlen = 2048
        self.batch_size = 1

args = Args()

# 加载原始模型和分词器

cache_dir = "/root/autodl-tmp/llm_weights"
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B",
    cache_dir=cache_dir,
    device_map="auto",  # 让 Hugging Face 自动分配多个 GPU
    torch_dtype=torch.float16
)

tokenizer_name = "HuggingFaceM4/llama-7b-tokenizer"
tokenizer = LlamaTokenizer.from_pretrained(tokenizer_name)



# 剪枝函数定义 --------------------------------------------------------------
def prune_linear_layer(layer, mask, dim):
    """结构化剪枝线性层"""
    if mask is None:
        return layer
    
    indices = mask.nonzero().squeeze()
    w = layer.weight.index_select(dim, indices)
    if layer.bias is not None and dim == 1:
        b = layer.bias
    else:
        b = None
    
    pruned_layer = nn.Linear(w.size(1), w.size(0), bias=layer.bias is not None)
    pruned_layer.weight.data.copy_(w)
    if b is not None:
        pruned_layer.bias.data.copy_(b)
    return pruned_layer

def compute_layer_importance(layer):
    """计算层重要性（适配GQA结构并分离梯度）"""
    scores = {}
    
    # 禁用梯度计算以节省内存
    with torch.no_grad():
        # 获取注意力头参数
        num_heads = layer.self_attn.num_heads
        num_key_value_heads = layer.self_attn.num_key_value_heads
        head_dim = layer.self_attn.head_dim

        # 计算查询头重要性（Q）
        q_proj = layer.self_attn.q_proj.weight.detach()
        q_heads = q_proj.view(-1, num_heads, head_dim)
        q_importance = torch.norm(q_heads, p=2, dim=(0,2))  # [num_heads]
        
        # 计算键值头重要性（K/V）
        k_proj = layer.self_attn.k_proj.weight.detach()
        k_heads = k_proj.view(-1, num_key_value_heads, head_dim)
        k_importance = torch.norm(k_heads, p=2, dim=(0,2))  # [num_kv_heads]
        
        v_proj = layer.self_attn.v_proj.weight.detach()
        v_heads = v_proj.view(-1, num_key_value_heads, head_dim)
        v_importance = torch.norm(v_heads, p=2, dim=(0,2))  # [num_kv_heads]
        
        # 扩展键值头重要性以匹配查询头
        group_size = num_heads // num_key_value_heads
        expanded_k = k_importance.repeat_interleave(group_size)
        expanded_v = v_importance.repeat_interleave(group_size)
        
        # 合并重要性（Q:50%, K:30%, V:20%）
        head_importance = q_importance*0.5 + expanded_k*0.3 + expanded_v*0.2

        # MLP门控结构重要性（保持原逻辑）
        gate = layer.mlp.gate_proj.weight.detach()
        up = layer.mlp.up_proj.weight.detach()
        down = layer.mlp.down_proj.weight.detach()
        gate_importance = torch.norm(gate, p=1, dim=0)
        updown_importance = (torch.norm(up, p=1, dim=0) + torch.norm(down, p=1, dim=1))/2
    
    return {
        'attention_heads': head_importance.cpu().numpy(),
        'gate_channels': gate_importance.cpu().numpy(),
        'updown_channels': updown_importance.cpu().numpy()
    }

def prune_model(model, target_sparsity=0.7):
    """执行全局剪枝（完整设备管理）"""
    device = next(model.parameters()).device  # 自动获取当前设备
    model.eval()
    
    # 计算各层重要性（保持原逻辑）
    layer_scores = {}
    for layer_id, layer in enumerate(model.model.layers):
        layer_scores[layer_id] = compute_layer_importance(layer)
    
    # 剪枝掩码生成（保持原逻辑）...
    
    # 应用剪枝
    for layer_id, layer in enumerate(model.model.layers):
        masks = prune_masks[layer_id]
        
        # 转换掩码到当前设备
        masks['heads'] = torch.tensor(masks['heads'], device=device)
        masks['gate'] = torch.tensor(masks['gate'], device=device)
        masks['updown'] = torch.tensor(masks['updown'], device=device)
        
        # 剪枝注意力头
        head_mask = ~torch.isin(
            torch.arange(layer.self_attn.num_heads, device=device),
            masks['heads']
        )
        
        if len(masks['heads']) > 0:
            layer.self_attn.num_heads -= len(masks['heads'])
            layer.self_attn.q_proj = prune_linear_layer(layer.self_attn.q_proj, head_mask, 0)
            layer.self_attn.k_proj = prune_linear_layer(layer.self_attn.k_proj, head_mask, 0)
            layer.self_attn.v_proj = prune_linear_layer(layer.self_attn.v_proj, head_mask, 0)
            layer.self_attn.o_proj = prune_linear_layer(layer.self_attn.o_proj, head_mask, 1)
        
        # 剪枝MLP层
        gate_mask = ~torch.isin(
            torch.arange(layer.mlp.gate_proj.out_features, device=device),
            masks['gate']
        )
        updown_mask = ~torch.isin(
            torch.arange(layer.mlp.up_proj.out_features, device=device),
            masks['updown']
        )
        
        layer.mlp.gate_proj = prune_linear_layer(layer.mlp.gate_proj, gate_mask, 1)
        layer.mlp.up_proj = prune_linear_layer(layer.mlp.up_proj, updown_mask, 1)
        layer.mlp.down_proj = prune_linear_layer(layer.mlp.down_proj, updown_mask, 0)
    
    return model

# 评估函数 ----------------------------------------------------------------
def get_loaders(dataset, tokenizer, seqlen=2048):
    """获取数据加载器"""
    if dataset == "wikitext2":
        testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")
    else:
        raise ValueError("Unsupported dataset")
    
    # 分割样本
    nsamples = testenc.input_ids.numel() // seqlen
    testenc = testenc.input_ids[:, :(nsamples*seqlen)]
    testenc = testenc.reshape(nsamples, seqlen)
    
    class DataLoader:
        def __init__(self, data, batch_size=1):
            self.data = data
            self.batch_size = batch_size
            self.num_batches = len(data) // batch_size
        
        def __iter__(self):
            for i in range(0, len(self.data), self.batch_size):
                yield self.data[i:i+self.batch_size]
    
    return None, DataLoader(testenc)

def eval_ppl_wikitext(model, testenc, bs=1, device="cuda"):
    """评估wikitext困惑度"""
    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.config.max_position_embeddings
    nlls = []
    
    progress_bar = tqdm(range(0, nsamples, bs), desc="Evaluating")
    for i in progress_bar:
        j = min(i+bs, nsamples)
        inputs = testenc[:, i*model.config.max_position_embeddings:j*model.config.max_position_embeddings]
        inputs = inputs.reshape(j-i, model.config.max_position_embeddings).to(device)
        
        with torch.no_grad():
            lm_logits = model(inputs).logits
        
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), 
                        shift_labels.reshape(-1))
        nlls.append(loss.float() * model.config.max_position_embeddings * (j-i))
    
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.config.max_position_embeddings))
    return ppl.item()

def eval_ppl(model, tokenizer):
    """完整评估流程"""
    _, test_loader = get_loaders(args.dataset, tokenizer, args.seqlen)
    model.eval()
    ppl = eval_ppl_wikitext(model, test_loader, args.batch_size)
    return ppl

# 执行流程 ----------------------------------------------------------------
if __name__ == "__main__":
    # Step 1: 剪枝模型
    print("开始剪枝...")
    pruned_model = prune_model(model, target_sparsity=0.7)
    
    # Step 2: 评估原始模型
    print("\n评估原始模型:")
    original_ppl = eval_ppl(model, tokenizer)
    print(f"原始模型困惑度: {original_ppl:.2f}")
    
    # Step 3: 评估剪枝后模型
    print("\n评估剪枝后模型:")
    pruned_ppl = eval_ppl(pruned_model, tokenizer)
    print(f"剪枝后困惑度: {pruned_ppl:.2f}")
    
    # Step 4: 保存结果
    print(f"\n最终结果:\n原始PPL: {original_ppl:.2f}\n剪枝后PPL: {pruned_ppl:.2f}")
    pruned_model.save_pretrained("llama3_8b_pruned_70percent")