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

def train():
    TOKENIZER_DIR = "dataset/processed/tab_tokenizer"
    DATA_DIR = "dataset/processed/individual"
    OUTPUT_DIR = "models/tiny-tab-v1"

    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)

    # 2. Load Model
    config = get_tiny_tab_config(vocab_size=len(tokenizer))
    model = T5ForConditionalGeneration(config)

    # Check for existing checkpoints
    last_checkpoint = None
    if os.path.exists(OUTPUT_DIR):
        checkpoints = [d for d in os.listdir(OUTPUT_DIR) if d.startswith("checkpoint")]
        if checkpoints:
            checkpoints.sort(key=lambda x: int(x.split("-")[1]))
            last_checkpoint = os.path.join(OUTPUT_DIR, checkpoints[-1])
            print(f"Resuming from checkpoint: {last_checkpoint}")

    # 3. Load Datasets
    train_dataset = TabDataset(
        "dataset/processed/train_files.txt",
        DATA_DIR,
        tokenizer
    )
    val_dataset = TabDataset(
        "dataset/processed/val_files.txt",
        DATA_DIR,
        tokenizer
    )

    # 4. Training Arguments
    # Optimized for hardware: 8GB VRAM (256-dim model fits comfortably)
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=2,
        eval_strategy="steps",
        eval_steps=250,
        save_steps=500,
        logging_steps=50,
        learning_rate=8e-4, # Higher for faster convergence
        num_train_epochs=10,
        weight_decay=0.01,
        warmup_steps=1000,
        fp16=True,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        save_total_limit=2,
        dataloader_num_workers=0, # Windows stability
    )

    # Optimizer
    optimizer = Adafactor(
        model.parameters(),
        lr=8e-4,
        relative_step=False,
        warmup_init=False
    )

    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        optimizers=(optimizer, None),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)] # Stop if no improvement in 3 evals
    )

    print("Starting training...")
    trainer.train(resume_from_checkpoint=last_checkpoint)

    print("Saving model...")
    model.save_pretrained(os.path.join(OUTPUT_DIR, "final"))
    tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final"))

if __name__ == "__main__":
    train()
