from datasets import load_from_disk
import os
try:
    print("Loading dataset...")
    dataset = load_from_disk("dataset/processed/cached_dataset")
    print("Success loading dataset")
    print(dataset)
except Exception as e:
    import traceback
    traceback.print_exc()
