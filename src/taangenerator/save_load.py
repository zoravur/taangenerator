import time
import os
import flax
import pickle
from flax.serialization import to_bytes, from_bytes


def save_model(params, directory="checkpoints"):
    os.makedirs(directory, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = os.path.join(directory, f"model_{timestamp}.pkl")

    with open(filename, "wb") as f:
        bytes_out = flax.serialization.to_bytes(params)
        pickle.dump(bytes_out, f)

    print(f"✔ saved model to {filename}")
    return filename


def load_model(path, template_params):
    with open(path, "rb") as f:
        return from_bytes(template_params, pickle.load(f))
