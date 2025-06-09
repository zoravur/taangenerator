import os
import random
import subprocess
from collections import defaultdict

from mido import Message, MidiFile, MidiTrack, MetaMessage
from pydub import AudioSegment

# === CONFIG ===
BPM = 200
BEATS = 12
BASE_MIDI_NOTE = 61

soundfont_path = os.path.expanduser("~/soundfonts/FluidR3_GM.sf2")
taal_path = os.path.expanduser("~/Documents/music/ektaal_200bpm_csharp.wav")

# === SWARA MAPPING ===
SWARA_DEGREES = {
    "S": 0, "R": 2, "G": 4, "M": 6, "P": 7, "D": 9, "N": 11,
}

def swara_to_note(swara: str) -> int:
    up = swara.count("'")
    down = swara.count("_")
    base = swara.strip("'_")
    if base not in SWARA_DEGREES:
        raise ValueError(f"Invalid swara: {swara}")
    return SWARA_DEGREES[base] + 12 * (up - down)

def note_to_swara(note: int) -> str:
    octave_offset, base_pitch = divmod(note, 12)
    inv = {v: k for k, v in SWARA_DEGREES.items()}
    if base_pitch not in inv:
        raise ValueError(f"Note {note} not mapped to swara")
    base = inv[base_pitch]
    if octave_offset == 0:
        return base
    elif octave_offset > 0:
        return base + ("'" * octave_offset)
    else:
        return ("_" * abs(octave_offset)) + base

notes = [-5, -3, -1, 0, 2, 4, 6, 7, 9, 11, 12, 14, 16]
note_names = {n: note_to_swara(n) for n in notes}
from_swaras = lambda swaras: [swara_to_note(s) for s in swaras]

# === MARKOV GENERATOR ===
class TaanGenerator:
    def __init__(self, notes, order=2):
        self.notes = notes
        self.order = order
        self.transitions = defaultdict(self._init_uniform)

    def _init_uniform(self):
        p = 1 / len(self.notes)
        return defaultdict(lambda: p, {n: p for n in self.notes})

    def observe(self, *history, next_note, w=1.0):
        if len(history) != self.order:
            raise ValueError(f"Expected {self.order} history elements")
        self.transitions[tuple(history)][next_note] += w

    def sample_next(self, history):
        history = tuple(history[-self.order:])
        options = self.transitions.get(history)
        if not options:
            return random.choice(self.notes)
        return random.choices(list(options), weights=list(options.values()))[0]

    def generate(self, beats, start_notes=None, end_note=None, tries=1000):
        if start_notes is None:
            start_notes = random.choices(self.notes, k=self.order)
        elif len(start_notes) != self.order:
            raise ValueError(f"start_notes must have {self.order} elements")
    
        for _ in range(tries):
            taan = list(start_notes)
            while len(taan) < beats * 2:
                next_note = self.sample_next(taan)
                taan.append(next_note)
            if end_note is None or taan[-1] == end_note:
                return taan
    
        return None  # failed to generate a valid taan


    def normalize(self):
        for key, targets in self.transitions.items():
            total = sum(targets.values())
            if total == 0:
                continue
            for k in targets:
                targets[k] /= total

    def decay_all(self, factor=0.99):
        for targets in self.transitions.values():
            for k in targets:
                targets[k] *= factor

# === DATASET BOOTSTRAPPING ===
def bootstrap(tg, phrases, weight=3.0):
    for phrase in phrases:
        for i in range(len(phrase) - tg.order):
            hist = tuple(phrase[i:i + tg.order])
            next_note = phrase[i + tg.order]
            tg.observe(*hist, next_note=next_note, w=weight)
    tg.normalize()

# === MIDI & AUDIO PIPELINE ===
def sequence_with_silence(taans, silence_beats, bpm=BPM):
    tick_per_note = 2  # 2 notes per beat
    silent = [None] * (silence_beats * tick_per_note)
    padded = []
    for taan in taans:
        padded += silent + taan
    return padded

def taan_to_midi(taan, path, bpm=BPM, base_note=BASE_MIDI_NOTE):
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)

    tempo = int(60_000_000 / bpm)
    track.append(MetaMessage("set_tempo", tempo=tempo, time=0))

    tick_per_beat = mid.ticks_per_beat
    tick = tick_per_beat // 2

    for note in taan:
        if note is None:
            track.append(Message("note_off", note=0, velocity=0, time=tick))
        else:
            pitch = base_note + note
            track.append(Message("note_on", note=pitch, velocity=100, time=0))
            track.append(Message("note_off", note=pitch, velocity=100, time=tick))

    mid.save(path)

def render_midi_to_wav(midi_path, wav_path, sf2_path):
    subprocess.run([
        "fluidsynth", "-g", "2.0", "-ni", sf2_path,
        midi_path, "-F", wav_path, "-r", "48000"
    ])

def mix_audio(taal_path, taan_path, out_path):
    taal = AudioSegment.from_wav(taal_path)
    taan = AudioSegment.from_wav(taan_path)
    taan = taan + 3  # volume boost
    taan = taan[:len(taal)]
    combined = taal.overlay(taan)
    combined.export(out_path, format="wav")

# === LEARNING LOOP ===
def reinforce(tg, taan, delta=0.05):
    for i in range(len(taan) - tg.order - 1):
        hist = tuple(taan[i:i + tg.order])
        next_note = taan[i + tg.order]
        tg.observe(*hist, next_note=next_note, w=delta)

def print_transition_matrix(tg, note_names=note_names, min_prob=0.01):
    print("\n=== TRANSITION MATRIX ===\n")
    for hist, nexts in sorted(tg.transitions.items()):
        hist_str = "(" + ", ".join(note_names.get(n, str(n)) for n in hist) + ") →"
        print(hist_str)
        for c, p in sorted(nexts.items(), key=lambda x: -x[1]):
            if p < min_prob:
                continue
            print(f"   {note_names.get(c, str(c)):>3}: {p:.2f} {'█'*int(p*20)}")
        print()

# === EXAMPLE PHRASES ===
yaman_seed_phrases = [
    from_swaras("_N R G M G R G M P D M P M D N R' N D N R' G' R' N R' S' N D P M P N N D P M P G M G R _N R S".split()),
    from_swaras("S' N D N S' R' S' N D N S' R' S' N D N S' G' R' S' N' S' R' R' S' N D P M P G M G R _N R S".split())
]

# === MAIN ===
# def main():
#     tg = TaanGenerator(notes, order=4)
#     bootstrap(tg, yaman_seed_phrases, weight=1.0)
# 
#     for round in range(10):
#         start = [-1, 2, 4, 6]      # S R
#         end = 0             # S
#         
#         t1 = tg.generate(BEATS, start_notes=start, end_note=end)
#         t2 = tg.generate(BEATS, start_notes=start, end_note=end)
# 
#         combined_seq = sequence_with_silence([t1, t2], silence_beats=12)
#         taan_to_midi(combined_seq, "taan_combined.mid")
#         render_midi_to_wav("taan_combined.mid", "taan_combined.wav", soundfont_path)
#         mix_audio(taal_path, "taan_combined.wav", "taan_mixed.wav")
# 
#         subprocess.run(["ffplay", "-nodisp", "-autoexit", "taan_mixed.wav"])
# 
#         vote = input("Which taan was better? [1/2]: ").strip()
#         if vote == "1":
#             reinforce(tg, t1)
#         elif vote == "2":
#             reinforce(tg, t2)
# 
#         tg.normalize()
#         print_transition_matrix(tg)


def main():
    tg = TaanGenerator(notes, order=4)
    bootstrap(tg, yaman_seed_phrases, weight=1.0)

    while True:
        start = [-1, 2, 4, 6]  # _N R G M
        end = 0               # S

        taan = tg.generate(BEATS, start_notes=start, end_note=end)

        if taan is None:
            print("Generation failed. Try relaxing constraints.")
            continue

        combined_seq = sequence_with_silence([taan], silence_beats=12)
        taan_to_midi(combined_seq, "taan_single.mid")
        render_midi_to_wav("taan_single.mid", "taan_single.wav", soundfont_path)
        mix_audio(taal_path, "taan_single.wav", "taan_mixed.wav")

        subprocess.run(["ffplay", "-nodisp", "-autoexit", "taan_mixed.wav"])

        try:
            delta = float(input("Reinforce delta (positive to reward, negative to punish): ").strip())/2
        except ValueError:
            print("Invalid input. Skipping.")
            continue

        reinforce(tg, taan, delta)
        tg.normalize()
        print_transition_matrix(tg)

if __name__ == "__main__":
    main()

