#!/bin/bash

# ==============================================================================
# Fretting-Transformer: End-to-End Pipeline
# ------------------------------------------------------------------------------
# This script executes the full pipeline from data download to model training.
# Ensure you have the requirements installed: pip install -r requirements.txt
# ==============================================================================

set -e # Exit on error

echo "🚀 Starting Fretting-Transformer Pipeline..."

# 1. Download Dataset
if [ ! -f "dataset/raw/data.zip" ]; then
    echo "📥 Step 1: Downloading vldsavelyev/guitar_tab dataset..."
    python scripts/00_download.py
else
    echo "⏭️ Step 1: Dataset already downloaded. Skipping."
fi

# 2. Process & Augment
if [ ! -d "dataset/processed/individual" ]; then
    echo "⚙️ Step 2: Processing AlphaTex files and applying augmentations..."
    rm -rf dataset/processed/*Individual* individuale individuale_individual
    python scripts/01_process.py
else
    echo "⏭️ Step 2: Data already processed. Skipping."
fi

# 3. Generate Vocabulary
if [ ! -f "dataset/processed/unique_tokens.json" ]; then
    echo "📖 Step 3: Generating technique-aware vocabulary..."
    python scripts/01.5_generate_vocab.py
else
    echo "📖 Step 3: Vocabulary already exists. Skipping."
fi

# 4. Tokenization
if [ ! -d "dataset/processed/tab_tokenizer" ]; then
    echo "🔢 Step 4: Tokenizing the dataset for T5..."
    python scripts/02_tokenize.py
else
    echo "⏭️ Step 4: Tokenizer already exists. Skipping."
fi

# 5. Split Dataset
if [ ! -f "dataset/processed/train_files.txt" ]; then
    echo "✂️ Step 5: Splitting into Train/Val/Test..."
    python scripts/03_split.py
else
    echo "⏭️ Step 5: Dataset split already exists. Skipping."
fi

# 6. Caching (Performance Step)
if [ ! -d "dataset/processed/cached_dataset" ]; then
    echo "📦 Step 6: Caching dataset for high-speed training..."
    python scripts/04_cache_dataset.py
else
    echo "⏭️ Step 6: Cached dataset already exists. Skipping."
fi

# 7. Training
echo "🎸 Step 7: Initializing Model Training..."
python scripts/train_model.py

echo "✅ Pipeline Complete! Check the 'models/' directory for checkpoints."
