# src/taangenerator/audio.py

import os
import subprocess
from dataclasses import dataclass
from mido import Message, MidiFile, MidiTrack, MetaMessage
from pydub import AudioSegment

BPM = 200


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


def play_audio(wav_path: str):
    subprocess.run(["ffplay", "-nodisp", "-autoexit", wav_path])
