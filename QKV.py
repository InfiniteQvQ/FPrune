import torch
from transformers import AutoModelForCausalLM, LlamaTokenizer
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

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
def compute_fisher_information(model, dataloader, device):
    fisher_info = {}
    model.eval()
    model.to(device)
    
    for input_ids, attention_mask in dataloader:
        input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
        
        outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss
        
        model.zero_grad()
        loss.backward()
        
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                fisher_value = param.grad ** 2  # Fisher Information: E[(∂L/∂w)²]
                if name not in fisher_info:
                    fisher_info[name] = fisher_value.detach().clone()
                else:
                    fisher_info[name] += fisher_value.detach().clone()
    
    # Normalize Fisher Information
    for name in fisher_info:
        fisher_info[name] /= len(dataloader)
    
    return fisher_info

# Run Fisher Information Computation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fisher_info = compute_fisher_information(model, dataloader, device)

# Visualize Fisher Information
def visualize_fisher(fisher_info):
    layer_names = list(fisher_info.keys())
    fisher_values = [torch.mean(fisher_info[name]).item() for name in layer_names]
    
    plt.figure(figsize=(12, 5))
    plt.barh(layer_names, fisher_values, color='blue')
    plt.xlabel("Fisher Information (Mean)")
    plt.ylabel("Layers")
    plt.title("Fisher Information for Each Layer in LLaMA 7B")
    plt.gca().invert_yaxis()
    plt.show()

visualize_fisher(fisher_info)
