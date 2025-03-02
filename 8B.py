import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict

class GradNormCalculator:
    def __init__(self, model_name: str, cache_dir: str = "/root/autodl-tmp/llm_weights"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print("[Init] Loading model:", model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            cache_dir=cache_dir,
            low_cpu_mem_usage=True,
            device_map="auto"  # 自动分配到多个GPU
        )
        
        print("[Init] Loading tokenizer:", model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.config = self.model.config
        self.gradients = {}

    def get_layers(self):
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers
        elif (hasattr(self.model, "model") 
              and hasattr(self.model.model, "decoder") 
              and hasattr(self.model.model.decoder, "layers")):
            return self.model.model.decoder.layers
        else:
            raise RuntimeError("无法定位到模型 decoder layers")

    def _hook_gradients(self, module, grad_in, grad_out, layer_idx: int):
        if grad_out[0] is not None:
            # 确保梯度转移到CPU，并转换为float32避免精度问题
            self.gradients[layer_idx] = grad_out[0].detach().to(device="cpu", dtype=torch.float32)

    def compute_gradnorm(self, num_samples: int = 8, seq_len: int = 128) -> Dict[int, float]:
        self.gradients.clear()
        layers = self.get_layers()

        # 注册梯度钩子
        hooks = []
        for idx, layer in enumerate(layers):
            h = layer.register_full_backward_hook(
                lambda m, gi, go, idx=idx: self._hook_gradients(m, gi, go, idx)
            hooks.append(h)

        self.model.train()

        # 生成输入数据（确保数据在主设备）
        input_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Artificial Intelligence is transforming the world.",
            "In a distant future, humans and AI coexist in harmony.",
            "OpenAI's ChatGPT demonstrates impressive reasoning capabilities."
        ] * (num_samples // 4 + 1)
        inputs = self.tokenizer(
            input_texts, 
            return_tensors="pt", 
            padding="max_length", 
            truncation=True, 
            max_length=seq_len
        )
        input_ids = inputs["input_ids"].to(self.model.device)  # 使用模型所在的主设备

        # 前向传播
        outputs = self.model(input_ids, labels=input_ids)
        loss = outputs.loss
        
        # 反向传播
        loss.backward()

        # 移除钩子
        for h in hooks:
            h.remove()
        torch.cuda.empty_cache()

        # 计算梯度范数
        gradnorm_per_layer = {}
        for layer_idx, grad in self.gradients.items():
            if grad is not None:
                grad_norm = torch.norm(grad).item()
                gradnorm_per_layer[layer_idx] = grad_norm

        return gradnorm_per_layer

if __name__ == "__main__":
    model_name = "meta-llama/Meta-Llama-3-8B"
    calculator = GradNormCalculator(model_name)

    print("[Step 1] 计算 GradNorm ...")
    gradnorm_results = calculator.compute_gradnorm(num_samples=8, seq_len=128)

    print("[GradNorm 结果]")
    for layer, norm in sorted(gradnorm_results.items()):
        print(f"Layer {layer:3d}: GradNorm = {norm:.6f}")