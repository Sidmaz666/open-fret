import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer

class TabDataset(Dataset):
    def __init__(self, file_list_path, data_dir, tokenizer, max_length=512):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        with open(file_list_path, 'r', encoding='utf-8') as f:
            self.file_names = f.read().splitlines()
        
    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.file_names[idx])
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data_list = json.load(f)
                # For now, we just take the first track/segment if multiple
                data = data_list[0]
                
                midi_text = " ".join(data['midi']) + " " + self.tokenizer.eos_token
                tab_text = " ".join(data['tab']) + " " + self.tokenizer.eos_token
                
                # Tokenize
                inputs = self.tokenizer(
                    midi_text, 
                    max_length=self.max_length, 
                    padding="max_length", 
                    truncation=True, 
                    return_tensors="pt"
                )
                
                labels = self.tokenizer(
                    tab_text, 
                    max_length=self.max_length, 
                    padding="max_length", 
                    truncation=True, 
                    return_tensors="pt"
                ).input_ids.squeeze()
                
                # Replace padding token id with -100 to ignore loss on padding
                labels[labels == self.tokenizer.pad_token_id] = -100
                
                return {
                    'input_ids': inputs.input_ids.squeeze(),
                    'attention_mask': inputs.attention_mask.squeeze(),
                    'labels': labels
                }
        except Exception as e:
            # Return a dummy item if error (or handle better)
            return self.__getitem__((idx + 1) % len(self))

def get_dataloader(split, data_dir, tokenizer_path, batch_size=8, max_length=512):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    file_list = f"dataset/processed/{split}_files.txt"
    dataset = TabDataset(file_list, data_dir, tokenizer, max_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"))

if __name__ == "__main__":
    # Test loading
    TOKENIZER_PATH = "dataset/processed/tokenizer.model"
    DATA_DIR = "dataset/processed/individual"
    if os.path.exists(TOKENIZER_PATH):
        try:
            loader = get_dataloader("val", DATA_DIR, "dataset/processed/", batch_size=2)
            batch = next(iter(loader))
            print("Batch keys:", batch.keys())
            print("Input shape:", batch['input_ids'].shape)
        except Exception as e:
            print(f"Error: {e}")
