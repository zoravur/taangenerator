from taangenerator.model import ModelParams, embedding_fn, stacked_forward
import jax.numpy as jnp
import numpy as np
import jax
from jax import lax
from functools import partial


def sample_autoregressively(
    params: ModelParams, start_tokens, max_len=32, temperature=1.0, top_k=None
):
    """
    start_tokens: (T,) array of initial token ids
    returns: (T + max_len,) array of generated token ids
    """
    generated = list(start_tokens)
    x = jnp.array(generated)[None, :]  # shape (1, T)

    for i in range(max_len):
        x_emb = embedding_fn(params.embedding, x)  # (1, T, D)
        x_out = stacked_forward(x_emb, params.transformer)  # (1, T, D)
        logits = x_out[:, -1, :] @ params.W_out.T  # (1, vocab)
        logits = logits / temperature

        if top_k is not None:
            top_logits = jnp.sort(logits, axis=-1)[:, -top_k]
            logits = jnp.where(logits < top_logits[:, None], -1e10, logits)

        probs = jax.nn.softmax(logits, axis=-1)
        next_token = jax.random.categorical(
            jax.random.PRNGKey(np.random.randint(1e6)), logits
        ).item()

        generated.append(next_token)
        x = jnp.array(generated)[None, :]

        # early stopping
        if ((i + 1) + len(start_tokens)) % 24 == 0 and next_token in [61, 73]:
            break

    return generated


@partial(jax.jit, static_argnames=["max_len", "top_k"])
def sample_with_scan(
    params,
    start_token,
    max_len,
    key,
    temperature=1.0,
    top_k=None,
):
    init_token = jnp.array(start_token)[None, :]  # (1, T)

    def step(carry, _):
        tokens, step_index, key = carry
        key, subkey = jax.random.split(key)

        x_emb = embedding_fn(params.embedding, tokens[:, :step_index])
        x_out = stacked_forward(x_emb, params.transformer)
        logits = x_out[:, -1, :] @ params.W_out.T
        logits = logits / temperature
        probs = jax.nn.softmax(logits, axis=-1)
        next_token = jax.random.categorical(subkey, logits).astype(jnp.int32)

        tokens = tokens.at[:, step_index].set(next_token)
        return (tokens, step_index + 1, key), None

    tokens = jnp.zeros((1, max_len), dtype=jnp.int32)
    tokens = tokens.at[:, 0].set(start_token[0])
    carry = (tokens, 1, key)

    (final_tokens, _, _), _ = jax.lax.scan(step, carry, None, length=max_len - 1)
    return final_tokens


from functools import partial
import jax
import jax.numpy as jnp
from jax import lax


@partial(jax.jit, static_argnames=["max_len", "top_k"])
def sample_with_while_loop(
    params, start_token, max_len, key, temperature=1.0, top_k=None
):
    tokens = jnp.zeros((1, max_len), dtype=jnp.int32)
    tokens = tokens.at[:, 0].set(start_token[0])
    step_index = jnp.array(1)

    def cond_fn(state):
        tokens, step_index, key = state
        return step_index < max_len

    def body_fn(state):
        tokens, step_index, key = state
        key, subkey = jax.random.split(key)

        # embed full tokens array up to step_index with masking
        x_emb = embedding_fn(params.embedding, tokens)  # (1, max_len, D)
        x_out = stacked_forward(x_emb, params.transformer)  # (1, max_len, D)
        last_hidden = x_out[:, step_index - 1, :]  # (1, D)

        logits = last_hidden @ params.W_out.T  # (1, vocab)
        logits = logits / temperature

        if top_k is not None:
            top_logits = jnp.sort(logits, axis=-1)[:, -top_k]
            logits = jnp.where(logits < top_logits[:, None], -1e10, logits)

        next_token = jax.random.categorical(subkey, logits).astype(jnp.int32)
        tokens = tokens.at[:, step_index].set(next_token)
        return (tokens, step_index + 1, key)

    tokens, _, _ = lax.while_loop(cond_fn, body_fn, (tokens, step_index, key))
    return tokens


def generate_looped_taan_sequence(n_taans, taan_generator, pad_token=None):
    """
    Generate a sequence of notes with taans and rests in between.
    Each taan: 24 notes
    Each rest: 24 steps (None)
    """
    all_notes = []
    for _ in range(n_taans):
        taan = taan_generator()
        assert (
            len(taan) % 24 == 0
        ), f"Taan generator must return multiple of 24 notes: {len(taan)} notes"
        all_notes.extend(taan)
        all_notes.extend([pad_token] * 24)
    return all_notes
