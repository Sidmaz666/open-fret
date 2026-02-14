import time
import torch
import os
import sys
import json
from transformers import T5ForConditionalGeneration, AutoTokenizer, Adafactor
from torch.utils.data import DataLoader, Dataset

# Add scripts to path for imports
sys.path.append(os.path.join(os.getcwd(), 'scripts'))

try:
    from config import get_tiny_tab_config
    from dataloader import TabDataset
except ImportError:
    # Minimal fallback for benchmark if scripts are missing
    def get_tiny_tab_config(vocab_size=4000):
        from transformers import T5Config
        return T5Config(
            vocab_size=vocab_size, 
            d_model=128, 
            d_ff=1024, 
            num_layers=3, 
            num_decoder_layers=3, 
            num_heads=4,
            is_encoder_decoder=True
        )

    class TabDataset(Dataset):
        def __init__(self, *args, **kwargs):
            self.file_names = ["dummy"] * 100
        def __len__(self): return 300053
        def __getitem__(self, idx):
            return {
                'input_ids': torch.zeros(512, dtype=torch.long),
                'attention_mask': torch.ones(512, dtype=torch.long),
                'labels': torch.zeros(512, dtype=torch.long)
            }

def estimate():
    TOKENIZER_DIR = "dataset/processed/tab_tokenizer"
    DATA_DIR = "dataset/processed/individual"
    TRAIN_FILES = "dataset/processed/train_files.txt"
    
    print("\n" + "="*50)
    print("      FRETTING-TRANSFORMER TRAINING ESTIMATOR")
    print("="*50)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Detecting hardware... Device: {device.upper()}")
    
    # 1. Load Tokenizer
    vocab_size = 32100 # Default T5
    tokenizer = None
    if os.path.exists(TOKENIZER_DIR):
        try:
            tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
            vocab_size = len(tokenizer)
            print(f"[*] Loaded tokenizer. Vocab size: {vocab_size}")
        except Exception as e:
            print(f"[!] Error loading tokenizer: {e}")
    else:
        print("[!] Tokenizer not found! Using default vocab size.")
    
    # 2. Load Model Configuration
    config = get_tiny_tab_config(vocab_size=vocab_size)
    print(f"[*] Initializing model architecture (T5 Small Variant)...")
    model = T5ForConditionalGeneration(config).to(device)
    
    # 3. Load Dataset or Estimate Size
    num_samples = 300053
    if os.path.exists(TRAIN_FILES) and tokenizer is not None:
        try:
            with open(TRAIN_FILES, 'r') as f:
                num_samples = len(f.read().splitlines())
            dataset = TabDataset(TRAIN_FILES, DATA_DIR, tokenizer, max_length=512)
            print(f"[*] Dataset size: {num_samples} samples (found train_files.txt)")
        except Exception as e:
            print(f"[!] Error loading actual dataset: {e}. Using synthetic data.")
            class DummyDataset(Dataset):
                def __len__(self): return num_samples
                def __getitem__(self, idx):
                    return {
                        'input_ids': torch.zeros(512, dtype=torch.long),
                        'attention_mask': torch.ones(512, dtype=torch.long),
                        'labels': torch.zeros(512, dtype=torch.long)
                    }
            dataset = DummyDataset()
    else:
        print(f"[*] Using synthetic dataset for benchmark. Size: {num_samples} samples.")
        class DummyDataset(Dataset):
            def __len__(self): return num_samples
            def __getitem__(self, idx):
                return {
                    'input_ids': torch.zeros(512, dtype=torch.long),
                    'attention_mask': torch.ones(512, dtype=torch.long),
                    'labels': torch.zeros(512, dtype=torch.long)
                }
        dataset = DummyDataset()

    batch_size = 16
    grad_acc = 4
    epochs = 50
    effective_batch_size = batch_size * grad_acc
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    optimizer = Adafactor(model.parameters(), lr=5e-4, relative_step=False, warmup_init=False)
    
    print(f"[*] Running performance benchmark (10 batches)...")
    
    model.train()
    
    # Bench batch
    input_ids = torch.zeros((batch_size, 512), dtype=torch.long).to(device)
    attention_mask = torch.ones((batch_size, 512), dtype=torch.long).to(device)
    labels = torch.zeros((batch_size, 512), dtype=torch.long).to(device)

    # Warmup
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss
    loss.backward()
    optimizer.zero_grad()
    
    start_time = time.time()
    num_benchmark_steps = 10
    
    for i in range(num_benchmark_steps):
        # Forward + Backward
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        
        if (i + 1) % grad_acc == 0:
            optimizer.step()
            optimizer.zero_grad()
            
    end_time = time.time()
    
    avg_time_per_batch = (end_time - start_time) / num_benchmark_steps
    print(f"[*] Performance: {avg_time_per_batch:.4f} seconds per batch")
    
    batches_per_epoch = num_samples // batch_size
    total_batches = batches_per_epoch * epochs
    
    total_time_seconds = total_batches * avg_time_per_batch
    total_time_hours = total_time_seconds / 3600
    total_time_days = total_time_hours / 24
    
    print("\n" + "="*50)
    print("              PROJECTION SUMMARY")
    print("="*50)
    print(f"Hardware:         {device.upper()}")
    print(f"Dataset Size:     {num_samples:,} samples")
    print(f"Training Epochs:  {epochs}")
    print(f"Effective BS:     {effective_batch_size}")
    print("-" * 50)
    print(f"Time per Batch:   {avg_time_per_batch:.4f}s")
    print(f"Time per Epoch:   {batches_per_epoch * avg_time_per_batch / 60:.2f} mins")
    print("-" * 50)
    print(f"TOTAL ESTIMATED:  {total_time_hours:.2f} hours")
    if total_time_hours > 24:
        print(f"                  ({total_time_days:.2f} days)")
    print("="*50)
    print("[!] Note: This estimate assumes constant load and no validation overhead.")
    print("[!] Early stopping or multi-GPU training will change these results.")
    print("="*50)

if __name__ == "__main__":
    estimate()
