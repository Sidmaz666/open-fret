import sys
import re

def tokens_to_alphatex(tokens_str, tempo=120, tuning=None):
    # Support both string input and list input
    if isinstance(tokens_str, str):
        tokens = tokens_str.split()
    else:
        tokens = tokens_str

    if tuning is None:
        tuning = "E5 B4 G4 D4 A3 E3"

    # Default T5 output tokens: TS_delta, TAB_string_fret
    # We need to reconstruct the AlphaTex format like :4 (1.1 2.2) 5.3 etc.
    
    alphatex_parts = []
    alphatex_parts.append(f"\\tempo {tempo}")
    alphatex_parts.append("\\track")
    alphatex_parts.append(f"\\tuning {tuning}")
    alphatex_parts.append(".")
    
    current_chord = []
    
    for t in tokens:
        if t.startswith("TS_"):
            # If we have a chord pending, format it
            if current_chord:
                alphatex_parts.append(format_chord(current_chord))
                current_chord = []
            
            delta = int(t.split("_")[1])
            # 480 is quarter note (:4)
            # Find the best duration token
            # We'll use a simple mapping for common durations
            dur_map = {
                1920: ":1",
                1440: ":1.",
                960: ":2",
                720: ":2.",
                480: ":4",
                360: ":4.",
                240: ":8",
                180: ":8.",
                120: ":16",
                90: ":16.",
                60: ":32",
                30: ":64"
            }
            # Also handle dotted notes
            # e.g. dotted quarter = 480 + 240 = 720
            # dotted eighth = 240 + 120 = 360
            
            # Simple heuristic for now
            if delta in dur_map:
                alphatex_parts.append(dur_map[delta])
            else:
                # Find closest standard or just use most frequent
                # AlphaTex allows :dur and then items
                # If delta is not a standard duration, we'll just skip for now or use r
                # This needs more complex logic to be robust
                pass
            
            # If delta is very large, it might be multiple measures/rests
            # but usually AlphaTex just does :4 r | if it's a rest
            # For simplicity, if it's a TS but no notes follows, it's a rest
            
        elif t.startswith("TAB_"):
            # TAB_string_fret_tech
            parts = t.split("_")
            if len(parts) >= 3:
                s = parts[1]
                f = parts[2]
                tech = parts[3] if len(parts) > 3 else ""
                current_chord.append(f"{f}.{s}{tech}")

    # Final chord
    if current_chord:
        alphatex_parts.append(format_chord(current_chord))

    return " ".join(alphatex_parts)

def format_chord(note_strs):
    if len(note_strs) == 1:
        return note_strs[0]
    else:
        return "(" + " ".join(note_strs) + ")"

def tokens_to_ascii(tokens_str):
    # Support both string input and list input
    if isinstance(tokens_str, str):
        tokens = tokens_str.split()
    else:
        tokens = tokens_str

    # 6 strings: e, B, G, D, A, E
    # Indices 1 to 6
    strings = {i: "" for i in range(1, 7)}
    
    current_pos = 0
    for t in tokens:
        if t.startswith("TS_"):
            try:
                # Add dashes to all strings based on duration
                # Scaling: TS_480 (quarter) -> 8 dashes
                delta = int(t.split("_")[1])
                dashes = max(1, delta // 60)
                for s in strings:
                    strings[s] += "-" * dashes
            except:
                pass
        elif t.startswith("TAB_"):
            # TAB_string_fret_tech
            parts = t.split("_")
            if len(parts) >= 3:
                s_idx = int(parts[1])
                fret = parts[2]
                tech = parts[3] if len(parts) > 3 else ""
                if s_idx in strings:
                    if strings[s_idx].endswith("-"):
                        strings[s_idx] = strings[s_idx][:-1] + fret + tech
                    else:
                        strings[s_idx] += fret + tech
        
    names = {1: "e", 2: "B", 3: "G", 4: "D", 5: "A", 6: "E"}
    output = []
    for i in range(1, 7):
        output.append(f"{names[i]}|{strings[i]}")
    
    return "\n".join(output)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Read from file or string
        content = sys.argv[1:]
        if "--tex" in content:
            content.remove("--tex")
            print(tokens_to_alphatex(" ".join(content)))
        else:
            print(tokens_to_ascii(" ".join(content)))
    else:
        # Read from stdin
        content = sys.stdin.read()
        print(tokens_to_ascii(content))
