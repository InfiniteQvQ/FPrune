import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict

class GradNormCalculator:
    def __init__(self, model_name: str, cache_dir: str = "/root/autodl-tmp/llm_weights"):
        """
        初始化 LLM，加载模型和 tokenizer，并准备计算 GradNorm。
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print("[Init] Loading model:", model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            cache_dir=cache_dir,
            low_cpu_mem_usage=True,
            device_map="auto"
        ).to(self.device)
        
        print("[Init] Loading tokenizer:", model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token  # 确保 padding token 存在
        
        self.config = self.model.config
        self.gradients = {}

    def get_layers(self):
        """
        获取 Transformer 的 Decoder 层，适用于 LLaMA、GPT-3.5、OPT 之类的自回归模型。
        """
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers
        elif (hasattr(self.model, "model") 
              and hasattr(self.model.model, "decoder") 
              and hasattr(self.model.model.decoder, "layers")):
            return self.model.model.decoder.layers
        else:
            raise RuntimeError("无法定位到模型 decoder layers，请根据实际结构修改 get_layers()")

    def _hook_gradients(self, module, grad_in, grad_out, layer_idx: int):
        """
        反向传播 Hook，存储每一层的梯度。
        """
        if grad_out[0] is not None:  # 确保梯度存在
            self.gradients[layer_idx] = grad_out[0].detach().cpu()

    def compute_gradnorm(self, num_samples: int = 8, seq_len: int = 128) -> Dict[int, float]:
        """
        计算 GradNorm，使用真实文本输入，遍历所有 Transformer 层。
        """
        self.gradients.clear()
        layers = self.get_layers()

        # 注册 Hook 以捕获梯度
        hooks = []
        for idx, layer in enumerate(layers):
            h = layer.register_full_backward_hook(lambda m, gi, go, idx=idx: self._hook_gradients(m, gi, go, idx))
            hooks.append(h)

        self.model.train()  # 训练模式，确保梯度计算

        # 生成真实文本输入
        input_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Artificial Intelligence is transforming the world.",
            "In a distant future, humans and AI coexist in harmony.",
            "OpenAI's ChatGPT demonstrates impressive reasoning capabilities."
        ]
        input_texts = input_texts * (num_samples // len(input_texts))  # 确保 num_samples 充足
        inputs = self.tokenizer(
            input_texts, return_tensors="pt", padding="max_length", truncation=True, max_length=seq_len
        )
        input_ids = inputs["input_ids"].to(self.device)

        labels = input_ids.clone()  # 复制 input_ids 作为训练目标
        outputs = self.model(input_ids, labels=labels)
        loss = outputs.loss
        loss.backward(retain_graph=True)  # 反向传播，确保梯度保留

        # 移除 Hook，防止影响后续训练
        for h in hooks:
            h.remove()
        torch.cuda.empty_cache()

        # 计算 GradNorm
        gradnorm_per_layer = {}
        for layer_idx, grad in self.gradients.items():
            grad_norm = torch.norm(grad).item()  # 计算 L2 范数
            gradnorm_per_layer[layer_idx] = grad_norm

        return gradnorm_per_layer

# -------------------- 执行 GradNorm 计算 --------------------

if __name__ == "__main__":
    model_name = "meta-llama/Meta-Llama-3-8B"  # 你可以替换成 LLaMA、GPT-3.5 或其他大模型
    calculator = GradNormCalculator(model_name)

    print("[Step 1] 计算 GradNorm ...")
    gradnorm_results = calculator.compute_gradnorm(num_samples=8, seq_len=128)

    print("[GradNorm 结果]")
    for layer, norm in gradnorm_results.items():
        print(f"Layer {layer}: GradNorm = {norm:.6f}")
