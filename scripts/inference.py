import torch
from transformers import T5ForConditionalGeneration, AutoTokenizer
import pretty_midi
import os
import argparse

def time_to_ticks(t, pm_res, target_res=480):
    return int(round(t * target_res * (2 if t > 0 else 0))) # This is wrong, let's use a better way

def get_ticks(pm, target_res=480):
    # Standard: bpm=120 -> 1 beat = 0.5s = target_res ticks
    # ticks = seconds * (bpm/60) * target_res
    # For simplicity, let's use pm.time_to_tick and scale by resolution
    pass

def midi_to_tokens(midi_path, target_res=480):
    pm = pretty_midi.PrettyMIDI(midi_path)
    res_scale = target_res / pm.resolution
    
    notes = []
    for instrument in pm.instruments:
        if not instrument.is_drum:
            notes.extend(instrument.notes)
    
    if not notes:
        return []

    notes.sort(key=lambda x: x.start)
    
    events = []
    for note in notes:
        start_tick = int(round(pm.time_to_tick(note.start) * res_scale))
        end_tick = int(round(pm.time_to_tick(note.end) * res_scale))
        events.append((start_tick, "ON", note.pitch))
        events.append((end_tick, "OFF", note.pitch))
    
    events.sort(key=lambda x: (x[0], 0 if x[1] == "OFF" else 1))
    
    tokens = []
    last_time = 0
    for time, type, pitch in events:
        delta = time - last_time
        if delta > 0:
            q_delta = int(round(delta))
            if q_delta > 0:
                tokens.append(f"TS_{q_delta}")
                last_time += q_delta
        
        if type == "ON":
            tokens.append(f"NO_{pitch}")
        else:
            tokens.append(f"NF_{pitch}")
            
    return tokens

def run_inference(midi_path, model_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = T5ForConditionalGeneration.from_pretrained(model_dir).to(device)
    model.eval()

    midi_tokens = midi_to_tokens(midi_path)
    if not midi_tokens:
        print("No notes found in MIDI.")
        return

    print(f"Input MIDI Tokens: {' '.join(midi_tokens[:30])}...")

    # ADDED CONDITIONING (from paper): Standard tuning and no capo
    # 64 59 55 50 45 40 is the standard guitar tuning MIDI numbers
    midi_text = " ".join(midi_tokens[:1024])
    conditioning = "CAPO_0 TUNING_64_59_55_50_45_40 "
    full_input = conditioning + midi_text
    
    inputs = tokenizer(full_input, return_tensors="pt", truncation=True, max_length=1024).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=512, # Enough for most transcriptions
            do_sample=True, # Sampling usually better for diversity
            top_p=0.9,
            temperature=0.7,
            num_return_sequences=1,
            early_stopping=True
        )

    decoded_tokens = tokenizer.decode(outputs[0], skip_special_tokens=True).split()
    
    # Paper Post-Processing: Neighbor Search
    try:
        from post_process import neighbor_search
        final_tokens = neighbor_search(decoded_tokens, midi_tokens)
        decoded = " ".join(final_tokens)
    except Exception as e:
        decoded = " ".join(decoded_tokens)
        
    return decoded

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--midi", type=str, required=True, help="Path to input MIDI file")
    parser.add_argument("--model", type=str, default="models/tiny-tab-v1/final", help="Path to trained model directory")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Error: Model directory not found: {args.model}")
        return

    result = run_inference(args.midi, args.model)
    
    # Clean up output tokens (only keep until first long repetitive sequence if any)
    # Actually model should have EOS.
    
    print("\n--- AlphaTex Output (.tex) ---")
    try:
        from tokens_to_tab import tokens_to_alphatex
        print(tokens_to_alphatex(result))
    except Exception as e:
        print(f"Error generating AlphaTex: {e}")

    print("\n--- ASCII Tablature (Preview) ---")
    try:
        from tokens_to_tab import tokens_to_ascii
        # Truncate to reasonable length if model looped
        ascii_tab = tokens_to_ascii(result)
        lines = ascii_tab.split("\n")
        print("\n".join([line[:100] for line in lines])) # Show first 100 chars
    except Exception as e:
        pass
    print("--------------------------------------")

if __name__ == "__main__":
    main()
