import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict

class GradNormCalculator:
    def __init__(self, model_name: str, cache_dir: str = "/root/autodl-tmp/llm_weights"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_gpus = torch.cuda.device_count()
        print("[Init] Available GPUs:", num_gpus)
        
        if num_gpus < 2:
            print("[Init] Less than 2 GPUs available, using auto device map.")
            device_map = "auto"
        else:
            print("[Init] Multiple GPUs detected, using custom device_map.")
            # 先加载到 CPU 上以获取模型结构
            model_temp = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                cache_dir=cache_dir,
                low_cpu_mem_usage=True,
                device_map="cpu"
            )
            
            # 判断模型结构：例如 Llama 模型可能是 model.layers 或 model.decoder.layers
            if hasattr(model_temp.model, "layers"):
                layers = model_temp.model.layers
                prefix = "model.layers"
            elif hasattr(model_temp.model, "decoder") and hasattr(model_temp.model.decoder, "layers"):
                layers = model_temp.model.decoder.layers
                prefix = "model.decoder.layers"
            else:
                raise RuntimeError("Cannot locate model decoder layers.")
            
            num_layers = len(layers)
            print(f"[Init] Model has {num_layers} layers. Assigning first half to cuda:0 and second half to cuda:1.")
            
            # 构造 device_map：先对各层进行划分
            device_map = {}
            for i in range(num_layers):
                device = "cuda:0" if i < num_layers // 2 else "cuda:1"
                device_map[f"{prefix}.{i}"] = device
            
            # 额外指定其他模块到设备上，避免未设置设备的问题
            device_map["model.embed_tokens"] = "cuda:0"  # 嵌入层
            if hasattr(model_temp.model, "norm"):
                device_map["model.norm"] = "cuda:0"
            elif hasattr(model_temp.model, "final_layer_norm"):
                device_map["model.final_layer_norm"] = "cuda:0"
            # 注意：对于 Llama 模型，lm_head 通常在顶层，而不是 model 下
            device_map["lm_head"] = "cuda:0"
            
            del model_temp  # 释放临时模型资源
        
        print("[Init] Loading model:", model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            cache_dir=cache_dir,
            low_cpu_mem_usage=True,
            device_map=device_map
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
            raise RuntimeError("Cannot locate model decoder layers.")
    
    def _hook_gradients(self, module, grad_in, grad_out, layer_idx: int):
        if grad_out[0] is not None:
            # 将梯度转移到 CPU，并转换为 float32，防止精度损失
            self.gradients[layer_idx] = grad_out[0].detach().to(device="cpu", dtype=torch.float32)
    
    def compute_gradnorm(self, num_samples: int = 8, seq_len: int = 128) -> Dict[int, float]:
        self.gradients.clear()
        layers = self.get_layers()
        
        # 为每一层注册反向传播钩子
        hooks = []
        for idx, layer in enumerate(layers):
            h = layer.register_full_backward_hook(
                lambda m, gi, go, idx=idx: self._hook_gradients(m, gi, go, idx)
            )
            hooks.append(h)
        
        self.model.train()
        
        # 生成输入数据
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
        input_ids = inputs["input_ids"].to(self.model.device)
        
        # 前向传播
        outputs = self.model(input_ids, labels=input_ids)
        loss = outputs.loss
        
        # 反向传播
        loss.backward()
        
        # 移除钩子并清理内存
        for h in hooks:
            h.remove()
        torch.cuda.empty_cache()
        
        # 计算每层梯度的范数
        gradnorm_per_layer = {}
        for layer_idx, grad in self.gradients.items():
            if grad is not None:
                grad_norm = torch.norm(grad).item()
                gradnorm_per_layer[layer_idx] = grad_norm
        
        return gradnorm_per_layer

if __name__ == "__main__":
    model_name = "meta-llama/Meta-Llama-3-8B"
    calculator = GradNormCalculator(model_name)
    
    print("[Step 1] Computing GradNorm ...")
    gradnorm_results = calculator.compute_gradnorm(num_samples=8, seq_len=128)
    
    print("[GradNorm Results]")
    for layer, norm in sorted(gradnorm_results.items()):
        print(f"Layer {layer:3d}: GradNorm = {norm:.6f}")
