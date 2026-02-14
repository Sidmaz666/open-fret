import os
import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict
import multiprocessing

def create_cache():
    TOKENIZER_DIR = "dataset/processed/tab_tokenizer"
    DATA_DIR = "dataset/processed/individual"
    TRAIN_LIST = "dataset/processed/train_files.txt"
    VAL_LIST = "dataset/processed/val_files.txt"
    CACHE_DIR = "dataset/processed/cached_dataset"
    MAX_LENGTH = 512

    print("🚀 Initializing Dataset Caching (6-8 hour target optimization)...")
    
    if not os.path.exists(TOKENIZER_DIR):
        print("❌ Tokenizer not found. Please run Step 3 (vocab generation) first.")
        return

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)

    def process_file_list(file_list_path):
        with open(file_list_path, 'r', encoding='utf-8') as f:
            file_names = f.read().splitlines()
        
        data_list = []
        print(f"📦 Loading and tokenizing {len(file_names)} files from {file_list_path}...")
        
        # We use a generator for 'datasets' to handle memory efficiently
        def gen():
            for fname in tqdm(file_names):
                fpath = os.path.join(DATA_DIR, fname)
                try:
                    with open(fpath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        
                        midi_text = " ".join(data['midi']) + " " + tokenizer.eos_token
                        tab_text = " ".join(data['tab']) + " " + tokenizer.eos_token
                        
                        # Note: We don't tokenize here yet if we want to use 'map', 
                        # but doing it here and saving to arrow is fastest for training.
                        yield {
                            "midi_text": midi_text,
                            "tab_text": tab_text
                        }
                except Exception:
                    continue

        return Dataset.from_generator(gen)

    # Create the datasets
    train_ds = process_file_list(TRAIN_LIST)
    val_ds = process_file_list(VAL_LIST)

    def tokenize_function(examples):
        # Using text_target is the modern way to tokenize labels in HuggingFace
        model_inputs = tokenizer(
            text=examples["midi_text"],
            text_target=examples["tab_text"],
            max_length=MAX_LENGTH, 
            truncation=True, 
            padding="max_length"
        )
        
        # Replace pad token with -100 in labels
        labels_ids = []
        for label_example in model_inputs["labels"]:
            labels_ids.append([(l if l != tokenizer.pad_token_id else -100) for l in label_example])
        
        model_inputs["labels"] = labels_ids
        return model_inputs

    print("🔢 Converting to model-ready tensors (Parallel Mapping)...")
    num_cpus = multiprocessing.cpu_count()
    
    tokenized_train = train_ds.map(
        tokenize_function, 
        batched=True, 
        num_proc=num_cpus if os.name != 'nt' else 1, # Windows multiproc safety
        remove_columns=["midi_text", "tab_text"]
    )
    
    tokenized_val = val_ds.map(
        tokenize_function, 
        batched=True, 
        num_proc=num_cpus if os.name != 'nt' else 1,
        remove_columns=["midi_text", "tab_text"]
    )

    ds_dict = DatasetDict({
        "train": tokenized_train,
        "validation": tokenized_val
    })

    print(f"💾 Saving cached dataset to {CACHE_DIR}...")
    ds_dict.save_to_disk(CACHE_DIR)
    print("✅ Caching Complete. Training will now be 10x faster.")

if __name__ == "__main__":
    create_cache()
