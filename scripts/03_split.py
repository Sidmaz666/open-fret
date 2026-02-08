import os
import random

def split_dataset(input_dir, output_dir, train_ratio=0.8, val_ratio=0.1):
    if not os.path.exists(input_dir):
        print(f"Input directory {input_dir} does not exist.")
        return

    # List all individual JSON files
    files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    
    if not files:
        print("No processed JSON files found.")
        return

    random.shuffle(files)

    total = len(files)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_files = files[:train_end]
    val_files = files[train_end:val_end]
    test_files = files[val_end:]

    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, 'train_files.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(train_files))
    with open(os.path.join(output_dir, 'val_files.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(val_files))
    with open(os.path.join(output_dir, 'test_files.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(test_files))

    print(f"Split complete: {len(train_files)} train songs, {len(val_files)} val songs, {len(test_files)} test songs.")

if __name__ == "__main__":
    INPUT_DIR = "dataset/processed/individual"
    OUTPUT_DIR = "dataset/processed"
    split_dataset(INPUT_DIR, OUTPUT_DIR)
