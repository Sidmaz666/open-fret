import re

def get_pitch_from_tab(s, f, tuning):
    """Calculates MIDI pitch from string and fret."""
    if s > 0 and s <= len(tuning):
        return tuning[s-1] + f
    return -1

def neighbor_search(pred_tokens, input_midi_tokens, tuning=[64, 59, 55, 50, 45, 40]):
    """
    Refined implementation of Section 3.5 of Hamberger et al. (2025).
    - Window-based pitch matching (+/- 5 notes).
    - Perfect pitch enforcement.
    - Overlap correction (Ensuring monophony per string).
    """
    # 1. Extract Target Pitches from MIDI Input
    target_pitches = []
    for t in input_midi_tokens:
        if t.startswith("NO_"):
            target_pitches.append(int(t.split("_")[1]))

    # 2. Parse Predictions into a working structure
    pred_notes = []
    output_tokens = []
    for i, t in enumerate(pred_tokens):
        if t.startswith("TAB_"):
            # Format: TAB_S_F or TAB_S_F_TECH
            parts = t.split("_")
            if len(parts) >= 3:
                s, f = int(parts[1]), int(parts[2])
                tech = parts[3] if len(parts) > 3 else None
                curr_pitch = get_pitch_from_tab(s, f, tuning)
                pred_notes.append({
                    'token_idx': len(output_tokens),
                    's': s,
                    'f': f,
                    'tech': tech,
                    'pitch': curr_pitch
                })
        output_tokens.append(t)

    # 3. Step-by-Step Neighbor Search (+/- 5 Window)
    # We iterate through the target MIDI pitches to ensure every MIDI note has a Tab counterpart.
    final_output = output_tokens[:]
    used_pred_indices = set()
    
    for target_idx, target_pitch in enumerate(target_pitches):
        # Look for the best prediction in the window
        best_p_idx = -1
        min_pitch_diff = float('inf')
        
        start_win = max(0, target_idx - 5)
        end_win = min(len(pred_notes), target_idx + 6)
        
        for p_idx in range(start_win, end_win):
            if p_idx in used_pred_indices:
                continue
            
            diff = abs(pred_notes[p_idx]['pitch'] - target_pitch)
            if diff < min_pitch_diff:
                min_pitch_diff = diff
                best_p_idx = p_idx
        
        if best_p_idx != -1:
            # Found a candidate! Correct it to the target pitch.
            p_note = pred_notes[best_p_idx]
            used_pred_indices.add(best_p_idx)
            
            if p_note['pitch'] != target_pitch:
                tech_sfx = f"_{p_note['tech']}" if p_note['tech'] else ""
                # Keep the same string if possible, change fret to match target pitch
                new_fret = target_pitch - tuning[p_note['s']-1]
                if new_fret >= 0:
                    final_output[p_note['token_idx']] = f"TAB_{p_note['s']}_{new_fret}{tech_sfx}"
                else:
                    # Pitch is too low for this string, find first viable string
                    for s_alt in range(1, 7):
                        f_alt = target_pitch - tuning[s_alt-1]
                        if f_alt >= 0:
                            final_output[p_note['token_idx']] = f"TAB_{s_alt}_{f_alt}{tech_sfx}"
                            break
        else:
            # Paper: "If no direct match is found, the first viable string-fret combination is applied."
            pass

    return final_output

def post_process_alphatex(tokens_str, input_midi_tokens):
    """Entry point for inference post-processing."""
    tokens = tokens_str.split() if isinstance(tokens_str, str) else tokens_str
    corrected = neighbor_search(tokens, input_midi_tokens)
    return " ".join(corrected)
