import os
import re
import json
import concurrent.futures
import multiprocessing
from tqdm import tqdm

def parse_tuning(tuning_str):
    # e.g., "E5 B4 G4 D4 A3 E3"
    notes = tuning_str.split()
    # Map note names to MIDI numbers
    note_map = {
        'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3, 'E': 4, 'F': 5, 
        'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8, 'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11
    }
    midi_numbers = []
    for note in notes:
        match = re.match(r"([A-G][#b]?)(-?\d+)", note)
        if match:
            name, octave = match.groups()
            midi = (int(octave) + 1) * 12 + note_map[name]
            midi_numbers.append(midi)
    return midi_numbers

def parse_alphatex(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception:
        return []

    # Simple regex based parser for alphaTex
    tempo = 120
    tempo_match = re.search(r'\\tempo (\d+)', content)
    if tempo_match:
        tempo = int(tempo_match.group(1))

    # Tracks
    tracks_raw = content.split('\\track')
    parsed_tracks = []

    for t_raw in tracks_raw[1:]: # Skip preamble
        tuning = [64, 59, 55, 50, 45, 40] # Default E-Standard
        tuning_match = re.search(r'\\tuning ([^\n\\]+)', t_raw)
        if tuning_match:
            tuning = parse_tuning(tuning_match.group(1).strip())

        current_duration = 4 # Default quarter note
        
        # Remove metadata lines and comments
        lines = t_raw.split('\n')
        clean_content = ""
        for line in lines:
            if not line.startswith('\\') and line.strip() and not line.strip() == '.':
                clean_content += " " + line

        # REMOVE EFFECTS BLOCKS {...} to ensure they don't break duration parsing
        clean_content = re.sub(r'\{[^\}]+\}', '', clean_content)

        # Tokenize
        pattern = r'(:[0-9]+\.*)|(\([^\)]+\)[.0-9\.]*)|([0-9]+\.[0-9]+[.0-9\.]*)|(r[.0-9\.]*)|(\|)'
        matches = re.finditer(pattern, clean_content)
        
        time_tokens = []
        current_ticks = 0
        ticks_per_quarter = 480
        
        for m in matches:
            token = m.group(0)
            if token.startswith(':'):
                val = token[1:].rstrip('.')
                dots = token.count('.')
                current_duration = int(val)
                continue
            if token == '|':
                continue
            
            dur = current_duration
            dots = 0
            
            dur_part_match = re.search(r'\.([0-9]+)(\.*)$', token)
            if dur_part_match:
                dur = int(dur_part_match.group(1))
                dots = len(dur_part_match.group(2))
            else:
                if token.endswith('.') and not token.startswith('('):
                    dots = token.count('.') - 1
                elif token.startswith('(') and token.endswith('.'):
                    dots = token.count('.')

            base_ticks = int(ticks_per_quarter * (4 / dur))
            actual_ticks = base_ticks
            multiplier = 0.5
            for _ in range(dots):
                actual_ticks += int(base_ticks * multiplier)
                multiplier *= 0.5
            
            notes_in_chord = []
            if token.startswith('('):
                note_matches = re.findall(r'([0-9]+)\.([0-9]+)', token)
                for f_str, s_str in note_matches:
                    notes_in_chord.append((int(f_str), int(s_str)))
            elif token.startswith('r'):
                pass
            else:
                note_match = re.match(r'([0-9]+)\.([0-9]+)', token)
                if note_match:
                    f_str, s_str = note_match.groups()
                    notes_in_chord.append((int(f_str), int(s_str)))

            if notes_in_chord:
                time_tokens.append({
                    'start': current_ticks,
                    'duration': actual_ticks,
                    'notes': notes_in_chord
                })
            current_ticks += actual_ticks

        parsed_tracks.append({
            'tuning': tuning,
            'tempo': tempo,
            'events': time_tokens
        })
        
    return parsed_tracks

def generate_sequences(parsed_track):
    midi_events = []
    tuning = parsed_track['tuning']
    events = parsed_track['events']
    
    for e in events:
        start = e['start']
        duration = e['duration']
        for fret, string in e['notes']:
            if string <= len(tuning):
                pitch = tuning[string-1] + fret
                midi_events.append((start, "ON", pitch, string, fret))
                midi_events.append((start + duration, "OFF", pitch, string, fret))
    
    midi_events.sort(key=lambda x: (x[0], 0 if x[1]=="OFF" else 1))
    
    midi_seq = []
    tab_seq = []
    last_time = 0
    for time, type, pitch, s, f in midi_events:
        delta = time - last_time
        if delta > 0:
            shift_token = f"TS_{delta}"
            midi_seq.append(shift_token)
            tab_seq.append(shift_token)
            last_time = time
        
        if type == "ON":
            midi_seq.append(f"NO_{pitch}")
            tab_seq.append(f"TAB_{s}_{f}")
        else:
            midi_seq.append(f"NF_{pitch}")
    
    return midi_seq, tab_seq

def process_file_single(file_path, output_dir_processed):
    """Processes one file and saves its result into a dedicated JSON file."""
    try:
        tracks = parse_alphatex(file_path)
        if not tracks: return 0
        
        file_name = os.path.basename(file_path)
        # Create a unique output name based on original path to avoid collisions
        # We'll use a hash or just the filename if it's unique enough.
        # Here we'll use the relative path parts joined by underscores.
        rel_path = os.path.relpath(file_path, "dataset/raw/extracted")
        out_name = rel_path.replace(os.sep, "__") + ".json"
        out_path = os.path.join(output_dir_processed, out_name)
        
        results = []
        for t in tracks:
            if not t['events']: continue
            m_seq, t_seq = generate_sequences(t)
            if m_seq and t_seq:
                results.append({
                    'midi': m_seq,
                    'tab': t_seq,
                    'file': file_name
                })
        
        if results:
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(results, f)
            return 1
    except Exception:
        pass
    return 0

def main():
    EXTRACTED_DIR = "dataset/raw/extracted"
    PROCESSED_DIR = "dataset/processed/individual"
    
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    print("Collecting files...")
    all_files = []
    for root, _, files in os.walk(EXTRACTED_DIR):
        for file in files:
            if file.endswith('.tex'):
                all_files.append(os.path.join(root, file))
    
    total_files = len(all_files)
    print(f"Total files found: {total_files}")
    
    # Using 75% of available cores to avoid complete CPU lockup
    num_subprocesses = max(1, int(multiprocessing.cpu_count() * 0.75))
    print(f"Processing using {num_subprocesses} workers (Optimal performance)...")
    
    success_count = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_subprocesses) as executor:
        # We pass PROCESSED_DIR to each call
        futures = {executor.submit(process_file_single, f, PROCESSED_DIR): f for f in all_files}
        
        # tqdm updates per file completion
        for future in tqdm(concurrent.futures.as_completed(futures), total=total_files, desc="Processing files", mininterval=0.5):
            success_count += future.result()

    print(f"\nFinished! Successfully processed {success_count} files.")
    print(f"Individual JSON files saved at: {PROCESSED_DIR}")

if __name__ == "__main__":
    main()
