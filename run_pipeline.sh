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
echo "📥 Step 1: Downloading vldsavelyev/guitar_tab dataset..."
python scripts/00_download.py

# 2. Process & Augment
# This applies the paper's techniques: Capo Augmentation (0-7) and Technique Parsing (Bends/Slides).
echo "⚙️ Step 2: Processing AlphaTex files and applying augmentations..."
# Clear old data if exists
rm -rf dataset/processed/*Individual* individuale individuale_individual
python scripts/01_process.py

# 3. Generate Vocabulary
# Captures the new TAB_S_F_TECH tokens for expressive guitar playing.
echo "📖 Step 3: Generating technique-aware vocabulary..."
python scripts/01.5_generate_vocab.py

# 4. Tokenization
echo "🔢 Step 4: Tokenizing the dataset for T5..."
python scripts/02_tokenize.py

# 5. Split Dataset
echo "✂️ Step 5: Splitting into Train/Val/Test..."
python scripts/03_split.py

# 6. Training
# Starts the T5-Small (reduced) training with paper-aligned hyperparams.
echo "🎸 Step 6: Initializing Model Training..."
python scripts/train_model.py

echo "✅ Pipeline Complete! Check the 'models/' directory for checkpoints."
