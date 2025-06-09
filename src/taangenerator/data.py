import os
import glob
import random
import numpy as np


def load_training_data_from_directory(path: str, seq_len: int) -> list[np.ndarray]:
    sequences = []
    for filename in glob.glob(f"{path}/*.txt"):
        sequences.extend(make_training_data_from_filename(filename, seq_len))
    return sequences


def batch_generator(sequences: list[np.ndarray], batch_size: int):
    while True:
        batch = random.sample(sequences, batch_size)
        batch = np.stack(batch, axis=0)  # (B, T)
        yield batch[:, :-1], batch[:, 1:]  # inputs, targets


def make_training_data_from_filename(
    filename, seq_len=8, vocab_size=88, filter_out_of_range=True
):
    """
    Reads a file containing space-separated integers, slices it into (seq_len + 1)-length windows,
    and returns training sequences for next-token prediction.

    Each sequence of length seq_len+1 is split into:
        input  = seq[:-1]
        target = seq[1:]

    Args:
        filename (str): path to the file containing space-separated note numbers
        seq_len (int): number of input tokens per training example (default: 8)
        vocab_size (int): maximum allowed token value (default: 88 for piano keys)
        filter_out_of_range (bool): whether to drop values outside [0, vocab_size-1]

    Returns:
        List[List[int]]: list of (seq_len + 1)-long sequences for training
    """
    with open(filename, "r") as f:
        text = f.read()

    tokens = list(map(int, text.strip().split()))

    if filter_out_of_range:
        tokens = [t for t in tokens if 0 <= t < vocab_size]

    data = []
    for i in range(len(tokens) - seq_len):
        seq = tokens[i : i + seq_len + 1]
        data.append(seq)

    return data


def make_training_data_from_directory(
    path="./data/*.txt", seq_len=8, vocab_size=88, filter_out_of_range=True
):
    """
    Reads all .txt files from the given directory glob, where each file contains
    space-separated note values. Generates training sequences independently per file.

    Returns:
        List[List[int]] – list of (seq_len + 1)-long sequences for training
    """
    all_sequences = []

    for filepath in glob.glob(path):
        with open(filepath, "r") as f:
            text = f.read()
        tokens = list(map(int, text.strip().split()))
        if filter_out_of_range:
            tokens = [t for t in tokens if 0 <= t < vocab_size]
        for i in range(len(tokens) - seq_len):
            seq = tokens[i : i + seq_len + 1]
            all_sequences.append(seq)

    return all_sequences
