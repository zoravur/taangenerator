import random
import subprocess
from taangenerator.sample import sample_autoregressively
from taangenerator.audio import notes_to_midi_file, render_midi_to_wav, mix_audio_looped
from taangenerator.utils import generate_looped_taan_sequence
from taangenerator.config import params, BPM, midi_path, taal_path


def one_taan():
    for _ in range(10):
        start_note = random.choice([60, 68, 73])
        taan = sample_autoregressively(
            params, [start_note], max_len=71, temperature=1.0
        )
        if taan[-1] in [61, 73]:
            return taan
    return taan  # fallback if no match


if __name__ == "__main__":
    n_taans = 10
    note_sequence = generate_looped_taan_sequence(n_taans, one_taan, pad_token=None)

    notes_to_midi_file(note_sequence, midi_path, bpm=BPM)
    render_midi_to_wav()

    mix_audio_looped(taal_path, "yaman.wav", "mixed.wav")
    subprocess.run(["ffplay", "-nodisp", "-autoexit", "mixed.wav"])
