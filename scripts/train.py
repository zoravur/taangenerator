# scripts/train.py

import jax
import jax.numpy as jnp
import numpy as np
from taangenerator.train import init_model_and_optimizer, train_step
from taangenerator.data import make_training_data_from_directory
from taangenerator.save_load import save_model
from taangenerator.config import batch_size, seq_len


def main():
    params, optimizer, opt_state = init_model_and_optimizer()

    sequences = make_training_data_from_directory("./data/*.txt", seq_len=seq_len)
    sequences = [np.array(seq, dtype=np.int32) for seq in sequences]

    n_steps = 10000

    for step in range(n_steps):
        batch = np.stack(
            [sequences[np.random.randint(len(sequences))] for _ in range(batch_size)]
        )
        inputs = jnp.array(batch[:, :-1])
        targets = jnp.array(batch[:, 1:])
        params, opt_state, loss = train_step(
            params, opt_state, inputs, targets, optimizer
        )
        if step % 100 == 0:
            print(f"[step {step}] loss: {loss:.4f}")

    save_model(params)


if __name__ == "__main__":
    main()
