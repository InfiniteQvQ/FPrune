import torch
from transformers import AutoModelForCausalLM, LlamaTokenizer
from torch.utils.data import DataLoader, Dataset
import numpy as np

# Load LLaMA 7B Model
cache_dir = "/root/autodl-tmp/llm_weights"
model = AutoModelForCausalLM.from_pretrained(
    "pinkmanlove/llama-7b-hf",
    cache_dir=cache_dir,
    device_map="auto",  # Automatically distribute across GPUs
    torch_dtype=torch.float16
)

tokenizer_name = "HuggingFaceM4/llama-7b-tokenizer"
tokenizer = LlamaTokenizer.from_pretrained(tokenizer_name)

# Sample Dataset (Replace with your own dataset)
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(self.texts[idx], max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")
        return encoding.input_ids.squeeze(0), encoding.attention_mask.squeeze(0)

# Example sentences (Replace with real dataset)
sentences = ["The quick brown fox jumps over the lazy dog.", "Artificial intelligence is revolutionizing the world."]
dataset = TextDataset(sentences, tokenizer)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Compute Fisher Information
def compute_fisher_information(model, dataloader, num_batches=10):
    fisher_info = {}
    model.eval()
    
    for batch_idx, (input_ids, attention_mask) in enumerate(dataloader):
        if batch_idx >= num_batches:
            break
        
        input_device = next(model.parameters()).device
        input_ids, attention_mask = input_ids.to(input_device), attention_mask.to(input_device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss
        
        model.zero_grad()
        loss.backward()
        
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                fisher_value = torch.mean(param.grad.float() ** 2).item()
                if name not in fisher_info:
                    fisher_info[name] = fisher_value
                else:
                    fisher_info[name] += fisher_value
    
    # Normalize Fisher Information
    for name in fisher_info:
        fisher_info[name] /= num_batches
    
    return fisher_info

# Run Fisher Information Computation
fisher_info = compute_fisher_information(model, dataloader)

# Sort Fisher Importance
sorted_layers = sorted(fisher_info.items(), key=lambda x: x[1], reverse=True)

# Print top 10 most important layers
print("Top 10 layers by Fisher importance:")
for layer, importance in sorted_layers[:]:
    print(f"{layer}: {importance:.6f}")
