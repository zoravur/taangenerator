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
    sample_autoregressively,
    sample_with_scan,
    sample_with_while_loop,
    generate_looped_taan_sequence,
)
from taangenerator.audio import (
    notes_to_midi_file,
    render_midi_to_wav,
    mix_audio,
    play_audio,
    AudioSegment,
)
from taangenerator.save_load import load_model


def latest_checkpoint(directory="checkpoints"):
    files = [
        f for f in os.listdir(directory) if f.startswith("model") and f.endswith(".pkl")
    ]
    if not files:
        raise FileNotFoundError("❌ no checkpoint found in 'checkpoints/'")
    return os.path.join(directory, sorted(files)[-1])


def one_taan(params):
    for _ in range(10):
        #     start = [random.choice([60, 68, 73])]
        key = jax.random.PRNGKey(int.from_bytes(os.urandom(4), "big"))
        taan = sample_with_while_loop(
            params,
            [random.choice([60, 68, 73])],
            max_len=72,
            key=key,
            temperature=1.0,
        )
        # taan = sample_autoregressively(
        #     params, [random.choice([60, 68, 73])], max_len=71, temperature=1.0
        # )
        # print(taan[0][-1])
        if taan[0][-1] in [61, 73]:
            return taan[0]
    return taan[0]  # fallback


def loop_taal_to_match(taal_path, ref_path, out_path):
    taal = AudioSegment.from_wav(taal_path)
    ref = AudioSegment.from_wav(ref_path)

    looped = taal * ((len(ref) // len(taal)) + 1)
    looped = looped[: len(ref)]
    looped.export(out_path, format="wav")
    return out_path


def main():
    print("📦 loading model...")
    dummy = ModelParams(
        embedding=init_embedding(jax.random.PRNGKey(0), vocab_size, d_model),
        transformer=init_stacked_transformer_params(
            jax.random.PRNGKey(1), d_model, num_layers
        ),
        W_out=jax.random.normal(jax.random.PRNGKey(2), (vocab_size, d_model)) * 0.01,
    )
    params = load_model(latest_checkpoint(), dummy)

    print(f"🎼 generating taan using model {latest_checkpoint()}...")
    # print(one_taan(params))

    note_sequence = generate_looped_taan_sequence(
        n_taans=10, taan_generator=lambda: one_taan(params).tolist()
    )

    print(note_sequence)

    midi_path = "yaman.mid"
    taan_path = "yaman.wav"
    taal_looped_path = "taal_looped.wav"
    out_path = "mixed.wav"

    notes_to_midi_file(note_sequence, midi_path)
    render_midi_to_wav(midi_path, taan_path, sf2_path=soundfont_path)
    loop_taal_to_match(taal_path, taan_path, taal_looped_path)
    mix_audio(taal_looped_path, taan_path, out_path)

    print("🔊 playing...")
    play_audio(out_path)


if __name__ == "__main__":
    main()
