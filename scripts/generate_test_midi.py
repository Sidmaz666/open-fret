import pretty_midi
import os

def create_test_midi(filename="test_guitar.mid"):
    # Create a PrettyMIDI object
    guitar_midi = pretty_midi.PrettyMIDI()
    
    # Create an Acoustic Guitar instrument (Program 25 or 26)
    guitar_program = pretty_midi.instrument_name_to_program('Acoustic Guitar (nylon)')
    guitar = pretty_midi.Instrument(program=guitar_program)

    # Let's create a simple C Major Arpeggio + Scale
    # Pitch, Start, End
    notes_to_add = [
        (48, 0, 0.5),   # C3
        (52, 0.5, 1.0), # E3
        (55, 1.0, 1.5), # G3
        (60, 1.5, 2.0), # C4
        (64, 2.0, 2.5), # E4
        (67, 2.5, 3.0), # G4
        (72, 3.0, 4.0), # C5 (Longer note)
        
        # A simple descending scale
        (71, 4.0, 4.5), # B4
        (69, 4.5, 5.0), # A4
        (67, 5.0, 5.5), # G4
        (65, 5.5, 6.0), # F4
        (64, 6.0, 7.0), # E4
    ]

    for pitch, start, end in notes_to_add:
        note = pretty_midi.Note(
            velocity=100,
            pitch=pitch,
            start=start,
            end=end
        )
        guitar.notes.append(note)

    # Add the guitar instrument to the PrettyMIDI object
    guitar_midi.instruments.append(guitar)

    # Write out the MIDI data
    guitar_midi.write(filename)
    print(f"✅ Successfully created test MIDI: {os.path.abspath(filename)}")

if __name__ == "__main__":
    create_test_midi()
