import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModelForCausalLM, LlamaTokenizer

# ------------------------------
# 1. 自定义一个 Pipeline 并行类
# ------------------------------
class PipelineParallelLlama(nn.Module):
    """
    将 LLaMA-7B 模型手动拆分到两个 GPU（前16层 on cuda:0，后16层 + norm/lm_head on cuda:1）。
    前向时会依次在 GPU0/1 上处理，并在中间显式 .to(device)。
    """
    def __init__(self, hf_model: AutoModelForCausalLM):
        super().__init__()
        # LLaMA 模型一般是 hf_model.model，
        # 其中包含 embed_tokens, layers[0..31], final_norm 等。
        self.embed_tokens = hf_model.model.embed_tokens
        self.layers = hf_model.model.layers
        self.norm = hf_model.model.norm
        self.lm_head = hf_model.lm_head
        self.config = hf_model.config  # 可能需要用到一些配置信息

        # 手动指定设备（假设你有两张 GPU:0 和 GPU:1）
        self.device_0 = torch.device("cuda:0")
        self.device_1 = torch.device("cuda:1")

        # 把相应的模块放到指定 GPU 上
        self.embed_tokens.to(self.device_0)
        # 前 16 层到 GPU 0
        for i in range(16):
            self.layers[i].to(self.device_0)
        # 后 16 层到 GPU 1
        for i in range(16, 32):
            self.layers[i].to(self.device_1)
        self.norm.to(self.device_1)
        self.lm_head.to(self.device_1)

    def forward(self, input_ids, attention_mask=None):
        """
        分两段来执行：
          1. GPU0: embed + 前16层
          2. 搬到 GPU1: 后16层 + norm + lm_head
        """
        # ---- GPU0 ----
        x = input_ids.to(self.device_0)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device_0)

        # (1) Embedding
        hidden_states = self.embed_tokens(x)

        # (2) 前16层
        for i in range(16):
            layer_module = self.layers[i]
            layer_outputs = layer_module(hidden_states, attention_mask=attention_mask)
            hidden_states = layer_outputs[0]  # LlamaDecoderLayer 返回的是 (hidden_states, None, ...)

        # 搬到 GPU1
        hidden_states = hidden_states.to(self.device_1)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device_1)

        # ---- GPU1 ----
        # (3) 后16层
        for i in range(16, 32):
            layer_module = self.layers[i]
            layer_outputs = layer_module(hidden_states, attention_mask=attention_mask)
            hidden_states = layer_outputs[0]

        # (4) 最终 norm + lm_head
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        return logits


# ------------------------------
# 2. 计算每层 QKV、Gate、Up、Down 等梯度 norm
# ------------------------------
def compute_grad_norms(model: PipelineParallelLlama):
    """
    遍历 LLaMA 模型中常见投影层的归一化梯度范数。
    返回一个 dict, 每个 key 对应一种投影层，每个 value 是 shape=[32] 的 ndarray（因为 32 层）。
    最后如果想要合并成 1D，可以自己再 flatten。
    """
    # LLaMA 中常见子模块名字
    module_keys = ["q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj"]

    num_layers = 32
    grad_norms = {key: np.zeros(num_layers) for key in module_keys}

    # nn.Module.named_parameters() 里，会包含 embed_tokens / layers.0 / ...
    # 每一层在 param 名字里一般像 "layers.0.self_attn.q_proj.weight" 这样
    named_params = dict(model.named_parameters())

    for name, param in named_params.items():
        if param.grad is None:
            continue
        # 找到类似 "layers.12.self_attn.q_proj" 这样的结构
        if "layers." not in name:
            continue
        # 提取第几层
        # name 大概是 "layers.12.xxx_proj.weight"
        # 先 split，然后找到第 1 个 "layers." 出现位置
        # 也可以用正则表达式，这里做个简单切分
        parts = name.split(".")
        # parts 可能是 ["layers", "12", "self_attn", "q_proj", "weight"]
        if len(parts) < 2:
            continue
        if not parts[1].isdigit():
            continue
        layer_idx = int(parts[1])
        # 看看是不是 q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/down_proj
        found_key = None
        for mk in module_keys:
            if mk in name:
                found_key = mk
                break
        if found_key is None:
            continue  # 不感兴趣的参数就跳过

        # 计算这个参数的 gradient norm
        g_norm = param.grad.data.norm(2).item()
        # 添加到 grad_norms
        grad_norms[found_key][layer_idx] += g_norm

    # 因为同一个层的 q_proj 里既有 weight 也可能有 bias，
    # 如果你也想把 bias 的 norm 加起来，这里可以再判断 "weight" / "bias" 合并。
    # 上面写的是相同 found_key 就 +=，因此 weight + bias 都会加到 grad_norms 上。

    # 对每个投影层做 max 归一化
    for key in grad_norms:
        max_val = np.max(grad_norms[key])
        if max_val > 0:
            grad_norms[key] /= max_val

    return grad_norms


# ------------------------------
# 3. 主流程示例
# ------------------------------
def main():
    # 1. 加载预训练的 LLaMA-7B
    cache_dir = "/root/autodl-tmp/llm_weights"
    model_name = "pinkmanlove/llama-7b-hf"
    print("Loading original HF LLaMA model...")

    # 由于是7B，最好用 fp16 来节省显存
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16
        cache_dir=cache_dir,
    )
    hf_model.gradient_checkpointing_enable()

    # 2. 将其包装成 Pipeline 并行模型
    pipeline_model = PipelineParallelLlama(hf_model)

    # 3. 加载分词器
    tokenizer_name = "HuggingFaceM4/llama-7b-tokenizer"
    tokenizer = LlamaTokenizer.from_pretrained(tokenizer_name)

    # 4. 构造一个简单的输入
    text = "Hello, this is a test input for importance computation."
    inputs = tokenizer(text, return_tensors="pt")

    # 5. 前向 + 反向传播
    #   注意：如果你要进一步节省显存，可以额外加 torch.cuda.amp.autocast() 等
    #   这里演示最简单的半精度计算
    with torch.autocast("cuda", dtype=torch.float16):
        logits = pipeline_model(**inputs)

    # 简单地用 mean(logits) 当作损失演示，实际可换成 cross_entropy
    loss = logits.mean()
    print(f"Loss: {loss.item():.4f}")

    loss.backward()

    # 6. 计算梯度范数
    grad_norms_dict = compute_grad_norms(pipeline_model)
    # grad_norms_dict 形如 {"q_proj": [32维], "k_proj": [32维], ...}

    # 如果你想合并成一维 (7个module_keys × 32层 = 224维)，可以自行 flatten
    # 先把 dict 转成一个大 list，再变成 numpy array
    all_norms = []
    for k in grad_norms_dict:
        all_norms.extend(grad_norms_dict[k])  # 直接拼接
    all_norms = np.array(all_norms)  # shape=(224,)

    print("Flattened grad norms shape:", all_norms.shape)
    print(all_norms)

    # 7. 释放显存，保存结果
    torch.cuda.empty_cache()
    np.save("llama_component_grad_norms.npy", grad_norms_dict)
    print("Gradient norm importance scores saved to llama_component_grad_norms.npy")


if __name__ == "__main__":
    main()
