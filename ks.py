import torch
import numpy as np
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, LlamaTokenizer

def compute_grad_norms(model):
    """
    计算 LLaMA 模型中常见投影层的归一化梯度范数。
    返回一个字典，每个键对应一种投影层，每个值是各层梯度范数的数组。
    """
    module_keys = ["q_proj", "k_proj", "v_proj", "o_proj", 
                   "gate_proj", "up_proj", "down_proj"]

    # LLaMA 的 TransformerBlock 在 model.model.layers 中
    model_layers = model.model.layers
    num_layers = len(model_layers)

    # 用 numpy 数组存梯度范数
    grad_norms = {key: np.zeros(num_layers) for key in module_keys}

    for i in range(num_layers):
        layer_key = f"model.layers.{i}"
        for key in module_keys:
            # 找到对应层、对应名字(key)的所有参数
            layer_params = [
                param.grad.detach().cpu()
                for name, param in model.named_parameters()
                if layer_key in name and key in name and param.grad is not None
            ]
            if layer_params:
                # 对该层所有相关参数的梯度范数取平均
                grad_norms[key][i] = np.mean([
                    torch.norm(p, p=2).item() for p in layer_params
                ])
            else:
                grad_norms[key][i] = 0.0

    # 对每个投影层的梯度做 max 归一化
    for key in grad_norms:
        max_val = np.max(grad_norms[key])
        if max_val > 0:
            grad_norms[key] /= max_val

    return grad_norms


def main():
    accelerator = Accelerator()  
    # 如果你在 accelerate config 中没有选择 fp16，可以在代码里写：
    # accelerator = Accelerator(mixed_precision="fp16")

    # 1. 加载 LLaMA-7B 模型（可以替换为你自己的模型名称或路径）
    model_name = "pinkmanlove/llama-7b-hf"
    tokenizer_name = "HuggingFaceM4/llama-7b-tokenizer"

    print("Loading model...")
    cache_dir = "/root/autodl-tmp/llm_weights"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        torch_dtype=torch.float16
    )

    # 启用 gradient checkpointing
    model.gradient_checkpointing_enable()

    # 2. 加载分词器
    print("Loading tokenizer...")
    tokenizer = LlamaTokenizer.from_pretrained(tokenizer_name)

    # 3. 用 accelerator.prepare 包装模型、优化器（若有）
    model = accelerator.prepare(model)

    # 4. 准备输入
    text = "Hello, this is a test input for importance computation."
    inputs = tokenizer(text, return_tensors="pt")

    # 把输入移动到对应 device
    inputs = {k: v.to(accelerator.device) for k, v in inputs.items()}

    # 5. 前向传播
    with accelerator.autocast():
        outputs = model(**inputs)

    # 6. 计算损失，这里仅示例直接对 logits 做 mean
    loss = outputs.logits.mean()
    print(f"Loss: {loss.item():.4f}")

    # 7. 反向传播
    accelerator.backward(loss)

    # 8. 计算梯度范数（注意：必须在 backward 之后）
    grad_norms = compute_grad_norms(model)

    # 9. 保存结果
    np.save("llama_component_grad_norms.npy", grad_norms)
    print("Gradient norm importance scores saved to llama_component_grad_norms.npy")


if __name__ == "__main__":
    main()
