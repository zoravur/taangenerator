import os

vocab_size = 88  # number of possible notes
seq_len = 24  # input length per example
d_model = 128  # hidden size
num_layers = 3  # transformer depth
batch_size = 32  # how many examples per step

BPM = 200


# print(str(os.path))
# 🪘 harmonium soundfont
soundfont_path = os.path.abspath("./assets/soundfonts/harmonium.sf2")

# 🕺 taal loop for mixing
taal_path = os.path.abspath("./assets/taals/ektaal_200bpm_csharp.wav")
