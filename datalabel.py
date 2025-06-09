import random
from collections import defaultdict
import pickle
import os

SAVE_PATH = "transitions.pkl"
NOTES = [-5, -3, -1, 0, 2, 4, 6, 7, 9, 11, 12, 14, 16]
SWARA_DEGREES = {"S": 0, "R": 2, "G": 4, "M": 6, "P": 7, "D": 9, "N": 11}
INV_SWARA = {v: k for k, v in SWARA_DEGREES.items()}

def note_to_swara(n):
    octave, pitch = divmod(n, 12)
    base = INV_SWARA.get(pitch, f"?{pitch}")
    if octave == 0: return base
    elif octave > 0: return base + ("'" * octave)
    else: return ("_" * -octave) + base

def pretty(seq):
    return " ".join(note_to_swara(n) for n in seq)

# === transition table ===
transitions = defaultdict(lambda: defaultdict(float))

# === load previous
if os.path.exists(SAVE_PATH):
    with open(SAVE_PATH, "rb") as f:
        transitions = pickle.load(f)
    print("loaded model.")
else:
    print("starting fresh.")

# === main labeling loop
try:
    while True:
        seq = tuple(random.choices(NOTES, k=5))

        # optional: reject overly repetitive
        if len(set(seq)) < 3:
            continue

        print("\nrate: ", pretty(seq), "→ ?", end=" ")

        try:
            raw = input().strip()
            if raw == "!":
                break
            weight = float(raw)
        except ValueError:
            print("skipped.")
            continue

        for n in NOTES:
            transitions[seq][n] += weight
except EOFError:
    print("\n[exit signal]")

# === normalize and save
for seq, dist in transitions.items():
    total = sum(dist.values())
    if total > 0:
        for n in dist:
            dist[n] /= total

with open(SAVE_PATH, "wb") as f:
    pickle.dump(transitions, f)

print("model saved to", SAVE_PATH)

