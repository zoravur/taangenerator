# scripts/demo.py

import os
import time
import random
import jax
import numpy as np

from taangenerator.config import *
from taangenerator.model import (
    ModelParams,
    init_embedding,
    init_stacked_transformer_params,
)
from taangenerator.sample import (
    sample_with_while_loop_batched,
)
from taangenerator.audio import (
    notes_to_midi_file,
    render_midi_to_wav,
    mix_audio,
    mix_audio_loop_longer,
    mix_audio_with_bpm_duration,
    mix_loop_to_main_length,
    play_audio,
    boost_audio,
    AudioSegment,
)
from taangenerator.save_load import load_model


# Alternative version with better efficiency
def generate_taans(params, n=10, max_attempts=50):
    """More efficient version with attempt limiting"""
    collected_taans = []
    attempts = 0

    while len(collected_taans) < n and attempts < max_attempts:
        attempts += 1

        # Generate random key
        key = jax.random.PRNGKey(int.from_bytes(os.urandom(4), "big"))

        # Generate batch - increase batch size if we need many more
        remaining_needed = n - len(collected_taans)
        batch_size = min(max(32, remaining_needed * 2), 128)  # Adaptive batch size

        sequences = sample_with_while_loop_batched(
            params,
            jax.random.randint(key, (batch_size,), 60, 80),
            max_len=72,
            key=key,
            batch_size=batch_size,
            temperature=1.0,
        )

        # Filter sequences ending in 61 or 73
        last_tokens = sequences[:, -1]
        mask = (last_tokens == 61) | (last_tokens == 73)
        filtered_sequences = sequences[mask]

        if len(filtered_sequences) > 0:
            collected_taans.extend([seq.tolist() for seq in filtered_sequences])
            print(
                f"Attempt {attempts}: Found {len(filtered_sequences)} valid sequences. "
                f"Total: {len(collected_taans)}/{n}"
            )
        else:
            print(f"Attempt {attempts}: No valid sequences found")

    if len(collected_taans) < n:
        print(
            f"Warning: Only collected {len(collected_taans)} out of {n} requested sequences"
        )

    return collected_taans[:n]


def generate_looped_taan_sequence(taans, pad_token=None):
    """
    Generate a sequence of notes with taans and rests in between.
    Each taan: 24 notes
    Each rest: 24 steps (None)
    """
    all_notes = []
    for taan in taans:
        assert (
            len(taan) % 24 == 0
        ), f"Taan generator must return multiple of 24 notes: {len(taan)} notes"
        all_notes.extend(taan)
        all_notes.extend([pad_token] * 24)
    return all_notes


def loop_taal_to_match(taal_path, ref_path, out_path):
    taal = AudioSegment.from_wav(taal_path)
    ref = AudioSegment.from_wav(ref_path)

    looped = taal * ((len(ref) // len(taal)) + 1)
    looped = looped[: len(ref)]
    looped.export(out_path, format="wav")
    return out_path


def main():
    yaman_checkpoint = "./checkpoints/model_20250609-090633.pkl"

    print("📦 loading model...")
    dummy = ModelParams(
        embedding=init_embedding(jax.random.PRNGKey(0), vocab_size, d_model),
        transformer=init_stacked_transformer_params(
            jax.random.PRNGKey(1), d_model, num_layers
        ),
        W_out=jax.random.normal(jax.random.PRNGKey(2), (vocab_size, d_model)) * 0.01,
    )
    params = load_model(yaman_checkpoint, dummy)

    print(f"🎼 generating taan using model {yaman_checkpoint}...")
    # print(one_taan(params))
    # generate_taans(params, n=10)

    note_sequence = generate_looped_taan_sequence(
        generate_taans(params, n=100), pad_token=None
    )

    print(len(note_sequence))

    # print(note_sequence)

    midi_path = "yaman.mid"
    taan_path = "yaman.wav"
    tabla_path = "./assets/taals/tabla_loop_220_bpm.wav"
    taanpura_path = "./assets/taals/taanpura_csharp.wav"
    out_path = "rendered.wav"

    notes_to_midi_file(note_sequence, midi_path)
    render_midi_to_wav(midi_path, taan_path, sf2_path=soundfont_path)
    # loop_taal_to_match(taal_path, taan_path, taal_looped_path)
    boost_audio(taan_path, taan_path, gain_db=-3.0)
    # boost_audio(tabla_path, tabla_path, gain_db=6.0)
    # boost_audio(taanpura_path, taanpura_path, gain_db=6.0)
    mix_audio_loop_longer(taan_path, taanpura_path, "temp.wav")
    # mix_audio_loop_longer("temp.wav", tabla_path, out_path)
    mix_loop_to_main_length(
        "temp.wav", tabla_path, out_path, bpm=220, beats_per_loop=12
    )

    print("🔊 mixing finished!")
    # print("🔊 playing...")
    # play_audio(out_path)


if __name__ == "__main__":
    main()
