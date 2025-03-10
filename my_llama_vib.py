# my_llama_vib.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaPreTrainedModel,
    LlamaAttention as OrigLlamaAttention,
    LlamaMLP as OrigLlamaMLP,
    LlamaDecoderLayer as OrigLlamaDecoderLayer,
    LlamaModel as OrigLlamaModel,
    LlamaForCausalLM as OrigLlamaForCausalLM,
    LlamaRMSNorm,
    rotate_half,
    apply_rotary_pos_emb,
    repeat_kv,
    _CONFIG_FOR_DOC,
    LlamaLinearScalingRotaryEmbedding,
    LlamaDynamicNTKScalingRotaryEmbedding,
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    # ...
)
from torch.nn import CrossEntropyLoss
import math
import time
from typing import Optional, Tuple, Union, List

# -------------------- 1) 定义可训练VIBMask (InformationBottleneck) --------------------
class InformationBottleneck(nn.Module):
    """
    一个最简化的 VIB:
      - mu, sigma (Parameter)
      - forward(x): 生成 mask 并乘上 x
      - kl_loss(): 计算 VIB KL
    """
    def __init__(self, size, kl_mult=1.0):
        super().__init__()
        self.size = size
        self.kl_mult = kl_mult
        self.mu = nn.Parameter(torch.zeros(size))
        self.sigma = nn.Parameter(torch.ones(size)*0.1)  # 初始值可自行调

    def forward(self, x):
        """
        x: shape [..., size], e.g. (batch, seq, num_heads) or (batch, seq, hidden_dim)
        """
        eps = torch.randn_like(self.sigma)
        mask = torch.sigmoid(self.mu + eps*self.sigma)
        # broadcast
        while mask.dim()<x.dim():
            mask = mask.unsqueeze(0)
        return x*mask

    def get_mask(self):
        eps = torch.randn_like(self.sigma)
        return torch.sigmoid(self.mu + eps*self.sigma)

    def kl_loss(self):
        # VIB KL
        kl = -0.5*torch.mean(1 + 2*self.sigma - self.mu.pow(2) - (2*self.sigma).exp())
        return kl*self.kl_mult

# -------------------- 2) 重写 LlamaAttention --------------------
class LlamaAttention(OrigLlamaAttention):
    """
    继承官方的 LlamaAttention 并在 __init__ 里插入 VIB (ib_q, ib_k, ib_v)
    """
    def __init__(self, config: LlamaConfig, layer_idx):
        super().__init__(config, layer_idx)
        if config.vib_layers:
            # 给 Q/K/V 各自一个 VIBMask
            self.ib_q = InformationBottleneck(self.num_heads, config.att_mul)
            self.ib_k = InformationBottleneck(self.num_key_value_heads, config.att_mul)
            self.ib_v = InformationBottleneck(self.num_key_value_heads, config.att_mul)
        else:
            self.ib_q = None
            self.ib_k = None
            self.ib_v = None

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
        hidden_mask=None,
    ):
        """
        覆盖官方 forward: 只在 Q/K/V 计算后，插入 VIB
        """
        # step1: 先跟官方写法一样: Q,K,V
        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1,2)
        key_states   = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1,2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1,2)

        # step2: Insert VIB
        if self.ib_q is not None:
            query_states = self.ib_q(query_states)
        if self.ib_k is not None:
            key_states = self.ib_k(key_states)
        if self.ib_v is not None:
            value_states = self.ib_v(value_states)

        # step3: 其余算子用 super() 处理 (不行的话就复制官方)
        # 但是 super().forward() 会重复 Q/K/V
        # => 你可以直接复制 remainder 过程 (rotary pos emb, scaled dot product, etc.)
        # 这里为了简单: 我直接 inline 剩下:
        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # ...
            pass  # 略, 同官方

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2,3))/math.sqrt(self.head_dim)
        if attention_mask is not None:
            attn_weights += attention_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)
        if attn_output.size()!=(bsz,self.num_heads,q_len,self.head_dim):
            raise ValueError("check shape")

        attn_output = attn_output.transpose(1,2).contiguous().view(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        if hidden_mask is not None:
            bsz_seq = bsz*q_len
            attn_output = attn_output.reshape(bsz_seq, self.hidden_size)
            attn_output = hidden_mask(attn_output)
            attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if not output_attentions:
            attn_weights=None
        return attn_output, attn_weights, past_key_value

    def vib_kl_loss(self):
        kl=0
        if self.ib_q: kl+=self.ib_q.kl_loss()
        if self.ib_k: kl+=self.ib_k.kl_loss()
        if self.ib_v: kl+=self.ib_v.kl_loss()
        return kl

# -------------------- 3) 重写 LlamaMLP --------------------
class LlamaMLP(OrigLlamaMLP):
    def __init__(self, config):
        super().__init__(config)
        if config.vib_layers:
            # gate, up, down
            self.ib_gate = InformationBottleneck(config.intermediate_size, config.inter_mul)
            self.ib_up   = InformationBottleneck(config.intermediate_size, config.inter_mul)
            self.ib_down = InformationBottleneck(config.hidden_size, config.inter_mul)
        else:
            self.ib_gate=None
            self.ib_up=None
            self.ib_down=None

    def forward(self, x, hidden_mask):
        bsz, seq, dim = x.size()
        if self.ib_gate is not None:
            # apply vib: gate, up
            g = torch.sigmoid(self.gate_proj(x))
            g = self.ib_gate(g)
            upv = self.up_proj(x)
            upv = self.ib_up(upv)
            out = g*upv
            out = self.down_proj(out)
            out = self.ib_down(out)

            # hidden_mask
            out = out.reshape(bsz*seq,dim)
            out = hidden_mask(out)
            out = out.reshape(bsz, seq, dim)
            return out
        else:
            # fallback original
            return super().forward(x, hidden_mask)

    def vib_kl_loss(self):
        kl=0
        if self.ib_gate:
            kl+=self.ib_gate.kl_loss()
        if self.ib_up:
            kl+=self.ib_up.kl_loss()
        if self.ib_down:
            kl+=self.ib_down.kl_loss()
        return kl

# -------------------- 4) 重写 LlamaDecoderLayer --------------------
class LlamaDecoderLayer(OrigLlamaDecoderLayer):
    """
    让 self.self_attn / self.mlp = 我们改写过的
    并加一个 vib_kl_loss()
    """
    def __init__(self, config, layer_idx=-1):
        super().__init__(config, layer_idx)
        # 替换
        self.self_attn = LlamaAttention(config, layer_idx)
        self.mlp = LlamaMLP(config)

    def vib_kl_loss(self, hidden_mask):
        return self.self_attn.vib_kl_loss() + self.mlp.vib_kl_loss()

# -------------------- 5) 重写 LlamaModel --------------------
class LlamaModel(OrigLlamaModel):
    """
    让 layers 用我们自定义 LlamaDecoderLayer
    并有 hidden_mask=InformationBottleneck(...) for embedding
    """
    def __init__(self, config):
        super().__init__(config)
        # 用 ModuleList 替换
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        if config.vib_layers:
            from vib_mask import InformationBottleneck
            self.hidden_mask = InformationBottleneck(config.hidden_size)
        else:
            self.hidden_mask = None

# -------------------- 6) 重写 VIBLlamaForCausalLM --------------------
class VIBLlamaForCausalLM(OrigLlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        # 替换 self.model = LlamaModel
        self.model = LlamaModel(config)
        # lm_head 不变

    def get_vib_kl_loss(self):
        # 遍历 self.model.layers
        kl=0
        for layer in self.model.layers:
            kl+=layer.vib_kl_loss(self.model.hidden_mask)
        return kl

# -------------------- 7) 辅助函数: freeze, ratio, etc. --------------------
def apply_vib_pruning_ratios(model, ratios):
    """
    model: VIBLlamaForCausalLM
    ratios: len = num_hidden_layers
    """
    layers = model.model.layers
    if len(ratios)!=model.config.num_hidden_layers:
        raise ValueError("layer mismatch")
    for i, layer in enumerate(layers):
        r = ratios[i]
        # self_attn
        if getattr(layer.self_attn, "ib_q", None):
            layer.self_attn.ib_q.sigma.data.fill_(r)
        if getattr(layer.self_attn, "ib_k", None):
            layer.self_attn.ib_k.sigma.data.fill_(r)
        if getattr(layer.self_attn, "ib_v", None):
            layer.self_attn.ib_v.sigma.data.fill_(r)
        # mlp
        if getattr(layer.mlp, "ib_gate", None):
            layer.mlp.ib_gate.sigma.data.fill_(r)
        if getattr(layer.mlp, "ib_up", None):
            layer.mlp.ib_up.sigma.data.fill_(r)
        if getattr(layer.mlp, "ib_down", None):
            layer.mlp.ib_down.sigma.data.fill_(r)

def freeze_non_mask_params(model):
    for name, param in model.named_parameters():
        # 如果带 "ib_" 或 "hidden_mask" 就可训练
        if ("ib_" in name) or ("hidden_mask" in name):
            param.requires_grad=True
        else:
            param.requires_grad=False

