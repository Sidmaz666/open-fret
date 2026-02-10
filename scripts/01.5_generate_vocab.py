import os
import json
from tqdm import tqdm

def generate_vocab():
    input_dir = "dataset/processed/individual"
    output_file = "dataset/processed/unique_tokens.json"
    
    if not os.path.exists(input_dir):
        print(f"Directory {input_dir} not found. Run 01_process.py first.")
        return

    files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    print(f"Collecting unique tokens from {len(files)} files...")
    
    unique_tokens = set()
    
    for filename in tqdm(files, desc="Scanning files"):
        file_path = os.path.join(input_dir, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Collect MIDI tokens
                if 'midi' in data:
                    unique_tokens.update(data['midi'])
                # Collect Tab tokens
                if 'tab' in data:
                    unique_tokens.update(data['tab'])
        except Exception as e:
            continue

    # Sort tokens for consistency
    sorted_tokens = sorted(list(unique_tokens))
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(sorted_tokens, f)
    
    print(f"Done! Saved {len(sorted_tokens)} unique tokens to {output_file}")

if __name__ == "__main__":
    generate_vocab()
