# src/taangenerator/audio.py

import os
import subprocess
from dataclasses import dataclass
from mido import Message, MidiFile, MidiTrack, MetaMessage
from pydub import AudioSegment
import math

BPM = 220


@dataclass
class SynthConfig:
    bpm: int = BPM
    velocity: int = 64
    duration: int = 480
    tempo: int = int(60_000_000 / BPM)
    tick_divisor: int = 2


def notes_to_midi_file(notes, filename: str, config: SynthConfig = SynthConfig()):
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)

    ticks = int(mid.ticks_per_beat / config.tick_divisor)
    track.append(MetaMessage("set_tempo", tempo=config.tempo, time=0))

    for note in notes:
        if note is None:
            track.append(Message("note_off", note=0, velocity=0, time=ticks))
        else:
            track.append(Message("note_on", note=note, velocity=100, time=0))
            track.append(Message("note_off", note=note, velocity=100, time=ticks))

    mid.save(filename)


def render_midi_to_wav(midi_path: str, wav_path: str, sf2_path: str):
    subprocess.run(
        [
            "fluidsynth",
            "-g",
            "2.0",
            "-ni",
            sf2_path,
            midi_path,
            "-F",
            wav_path,
            "-r",
            "48000",
        ]
    )


def mix_audio(taal_path: str, taan_path: str, out_path: str):
    taal = AudioSegment.from_wav(taal_path)
    taan = AudioSegment.from_wav(taan_path)

    taan = taan - 2
    taan = taan[: len(taal)]

    combined = taal.overlay(taan)
    combined.export(out_path, format="wav")


def boost_audio(audio_path: str, out_path: str, gain_db: float = 6.0):
    audio = AudioSegment.from_wav(audio_path)
    boosted_audio = audio + gain_db
    boosted_audio.export(out_path, format="wav")


def bpm_to_ms(bpm, beats, sample_rate=48000):
    """Convert BPM + beats to milliseconds"""
    samples_per_beat = (60 * sample_rate) / bpm
    total_samples = beats * samples_per_beat
    return int(round(total_samples))


def mix_audio_loop_longer(file1: str, file2: str, out_path: str):
    audio1 = AudioSegment.from_wav(file1)
    audio2 = AudioSegment.from_wav(file2)

    # Loop the shorter audio to match the length of the longer one
    if len(audio1) > len(audio2):
        audio2 = audio2 * ((len(audio1) // len(audio2)) + 1)
        audio2 = audio2[: len(audio1)]
    else:
        audio1 = audio1 * ((len(audio2) // len(audio1)) + 1)
        audio1 = audio1[: len(audio2)]

    combined = audio1.overlay(audio2)
    combined.export(out_path, format="wav")


def bpm_to_samples(bpm: float, beats: float, sample_rate: int = 48000) -> int:
    """Calculate exact samples for given BPM and beats"""
    return int(round((beats * 60 * sample_rate) / bpm))


def samples_to_ms(samples: int, sample_rate: int = 48000) -> float:
    """Convert samples to milliseconds"""
    return (samples / sample_rate) * 1000


def mix_audio_with_bpm_duration(
    file1: str,
    file2: str,
    out_path: str,
    bpm: float,
    beats: float,
    sample_rate: int = 48000,
):
    """Mix audio files with audio2 looped to exact BPM duration"""
    audio1 = AudioSegment.from_wav(file1)
    audio2 = AudioSegment.from_wav(file2)

    target_ms = samples_to_ms(bpm_to_samples(bpm, beats, sample_rate), sample_rate)

    # Loop and trim audio2 to target duration
    if len(audio2) < target_ms:
        audio2 = audio2 * math.ceil(target_ms / len(audio2))
    audio2 = audio2[:target_ms]

    # Pad or trim audio1 to match
    if len(audio1) < target_ms:
        audio1 = audio1 + AudioSegment.silent(duration=target_ms - len(audio1))
    else:
        audio1 = audio1[:target_ms]

    audio1.overlay(audio2).export(out_path, format="wav")


def mix_loop_to_main_length(
    main_file: str,
    loop_file: str,
    out_path: str,
    bpm: float,
    beats_per_loop: float = 16,
    sample_rate: int = 48000,
):
    """Loop backing track to match main audio length, rounded to complete loops"""
    main = AudioSegment.from_wav(main_file)
    loop = AudioSegment.from_wav(loop_file)

    # Calculate how many complete loops fit in main audio
    main_samples = int(round((len(main) / 1000) * sample_rate))
    samples_per_loop = bpm_to_samples(bpm, beats_per_loop, sample_rate)
    complete_loops = round(main_samples / samples_per_loop)

    # Calculate exact target duration
    target_ms = samples_to_ms(complete_loops * samples_per_loop, sample_rate)

    # Adjust both audios to target duration
    main = main[:target_ms]
    loop = (loop * math.ceil(target_ms / len(loop)))[:target_ms]

    main.overlay(loop).export(out_path, format="wav")


def play_audio(wav_path: str):
    subprocess.run(["ffplay", "-nodisp", "-autoexit", wav_path])
