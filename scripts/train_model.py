import os
import torch
from transformers import (
    T5ForConditionalGeneration,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    Adafactor,
    EarlyStoppingCallback
)
import sys
# Ensure standard imports work when running as script
try:
    from config import get_tiny_tab_config
    from dataloader import TabDataset
except ImportError:
    # If running from root as 'python scripts/train_model.py'
    # we need to add local dir
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from config import get_tiny_tab_config
    from dataloader import TabDataset

import multiprocessing
from transformers import TrainerCallback

class TrainLoggerCallback(TrainerCallback):
    def __init__(self, log_path):
        self.log_path = log_path
        with open(self.log_path, "a") as f:
            f.write("\n" + "="*40 + "\n")
            f.write("RESUMING LOG SESSION\n")
            f.write("="*40 + "\n")

    def on_epoch_end(self, args, state, control, **kwargs):
        # Professional summary at end of each epoch
        epoch = round(state.epoch)
        with open(self.log_path, "a") as f:
            f.write(f"\n--- Epoch {epoch} Summary ---\n")
            if state.log_history:
                # Find latest evaluation metrics
                eval_metrics = [h for h in state.log_history if "eval_loss" in h]
                if eval_metrics:
                    latest_eval = eval_metrics[-1]
                    f.write(f"Validation Loss: {latest_eval['eval_loss']:.4f}\n")
            f.write("-" * 20 + "\n")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            with open(self.log_path, "a") as f:
                log_str = f"Step: {state.global_step:06d} | "
                if "loss" in logs:
                    log_str += f"Train Loss: {logs['loss']:.4f} | "
                if "eval_loss" in logs:
                    log_str += f"Eval Loss: {logs['eval_loss']:.4f} | "
                if "learning_rate" in logs:
                    log_str += f"LR: {logs['learning_rate']:.2e} | "
                f.write(log_str + "\n")

from datasets import load_from_disk

def train():
    TOKENIZER_DIR = "dataset/processed/tab_tokenizer"
    CACHE_DIR = "dataset/processed/cached_dataset"
    OUTPUT_DIR = "models/tiny-tab-v1"
    LOG_FILE = "train_log.txt"

    # Determine Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"System Check: Detected {device.upper()} for training.")

    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)

    # 2. Load Model (Paper Architecture)
    config = get_tiny_tab_config(vocab_size=len(tokenizer))
    model = T5ForConditionalGeneration(config)

    # 3. Load Cached Datasets (Turbo speed optimization)
    if os.path.exists(CACHE_DIR):
        print(f"Loading high-speed cached dataset from {CACHE_DIR}")
        dataset = load_from_disk(CACHE_DIR)
        train_dataset = dataset["train"]
        val_dataset = dataset["validation"]
    else:
        print(f"Warning: Cache not found at {CACHE_DIR}. Falling back to slow file-by-file loading.")
        DATA_DIR = "dataset/processed/individual"
        train_dataset = TabDataset("dataset/processed/train_files.txt", DATA_DIR, tokenizer, max_length=512)
        val_dataset = TabDataset("dataset/processed/val_files.txt", DATA_DIR, tokenizer, max_length=512)

    # 4. Professional Training Arguments (Optimized for 6-8h target)
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=2,
        num_train_epochs=60, # Increased to 60 for better convergence
        learning_rate=5e-4, 
        weight_decay=0.01,
        warmup_steps=1000,
        lr_scheduler_type="cosine",
        eval_strategy="epoch", 
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        save_total_limit=2,
        ddp_find_unused_parameters=False,
    )

    # Optimizer: Adafactor
    optimizer = Adafactor(
        model.parameters(),
        lr=5e-4,
        relative_step=False,
        warmup_init=False,
    )

    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        optimizers=(optimizer, None),
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=10), # Increased patience for deeper refinement
            TrainLoggerCallback(LOG_FILE)
        ]
    )

    print(f"Executing Training Protocol: Resuming for 60 Epochs Total")
    with open(LOG_FILE, "a") as f:
        f.write(f"\nRESUMING TRAINING | Target Epochs: 60\n")
        f.write("-" * 40 + "\n")

    # Resume from latest checkpoint
    trainer.train(resume_from_checkpoint=True)

    print("Protocol Complete. Saving Optimized Model Artifacts...")
    model.save_pretrained(os.path.join(OUTPUT_DIR, "final"))
    tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final"))

if __name__ == "__main__":
    train()
