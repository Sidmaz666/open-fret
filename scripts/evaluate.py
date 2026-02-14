import torch
from transformers import T5ForConditionalGeneration, AutoTokenizer
from datasets import load_from_disk
import os
from tqdm import tqdm

def evaluate_model(model_path, dataset_path, num_samples=100):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Evaluating on {device}...")
    
    model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    dataset = load_from_disk(dataset_path)
    test_data = dataset["test"] if "test" in dataset else dataset["train"].select(range(num_samples))
    test_data = test_data.select(range(min(num_samples, len(test_data))))
    
    correct = 0
    total = 0
    
    all_raw_tokens = []
    
    for i in tqdm(range(len(test_data))):
        sample = test_data[i]
        input_ids = torch.tensor([sample["input_ids"]]).to(device)
        labels = sample["labels"]
        # Replace -100 with pad_token_id for decoding
        clean_labels = [l if l != -100 else 0 for l in labels]
        
        with torch.no_grad():
            outputs = model.generate(input_ids, max_new_tokens=256)
            
        pred_text = tokenizer.decode(outputs[0], skip_special_tokens=True).split()
        label_text = tokenizer.decode(clean_labels, skip_special_tokens=True).split()
        
        # Simple token accuracy (exact match)
        for p, l in zip(pred_text, label_text):
            if p == l:
                correct += 1
            total += 1
            
        all_raw_tokens.extend(pred_text[:5])
        
    accuracy = correct / total if total > 0 else 0
    print(f"\nExact Token Accuracy: {accuracy:.4f}")
    
    # Check for repetition
    from collections import Counter
    counts = Counter(all_raw_tokens)
    print("Most frequent predicted tokens:", counts.most_common(5))

if __name__ == "__main__":
    evaluate_model("models/tiny-tab-v1/final", "dataset/processed/cached_dataset")
