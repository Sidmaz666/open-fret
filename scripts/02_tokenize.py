from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast
import json
import os

def build_tokenizer():
    unique_tokens_file = "dataset/processed/unique_tokens.json"
    if not os.path.exists(unique_tokens_file):
        print("unique_tokens.json not found.")
        return

    with open(unique_tokens_file, 'r') as f:
        unique_tokens = json.load(f)

    # Standard specials
    specials = ["[PAD]", "[UNK]", "[BOS]", "[EOS]", "[MASK]"]
    vocab = specials + unique_tokens
    w2i = {word: i for i, word in enumerate(vocab)}

    # WordLevel model ensures each token is treated as an unbreakable unit
    tokenizer = Tokenizer(models.WordLevel(vocab=w2i, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    
    # Export to transformers format
    save_path = "dataset/processed/tokenizer.json"
    tokenizer.save(save_path)
    
    # Now wrap it in PreTrainedTokenizerFast for T5 compatibility
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=save_path,
        bos_token="[BOS]",
        eos_token="[EOS]",
        unk_token="[UNK]",
        pad_token="[PAD]",
        mask_token="[MASK]",
    )
    
    save_dir = "dataset/processed/tab_tokenizer"
    os.makedirs(save_dir, exist_ok=True)
    fast_tokenizer.save_pretrained(save_dir)
    print(f"Tokenizer (WordLevel) saved to {save_dir}")

    # Verify
    test_str = "TS_480 NO_64 [EOS]"
    encoded = fast_tokenizer.encode(test_str)
    print(f"Test encode: {test_str} -> {encoded}")
    decoded = fast_tokenizer.decode(encoded)
    print(f"Test decode: {encoded} -> {decoded}")
    print(f"Vocab size: {len(fast_tokenizer)}")

if __name__ == "__main__":
    build_tokenizer()
