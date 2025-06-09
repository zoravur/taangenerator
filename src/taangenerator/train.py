# taangenerator/train.py
from functools import partial

from taangenerator.config import *
from taangenerator.model import (
    ModelParams,
    init_stacked_transformer_params,
    init_embedding,
    forward_and_loss,
)
import jax
import optax


def init_model_and_optimizer(seed=0):
    key = jax.random.PRNGKey(seed)
    k1, k2, k3 = jax.random.split(key, 3)

    embedding = init_embedding(k1, vocab_size, d_model)
    transformer = init_stacked_transformer_params(k2, d_model, num_layers)
    W_out = jax.random.normal(k3, (vocab_size, d_model)) * 0.01

    params = ModelParams(embedding=embedding, transformer=transformer, W_out=W_out)

    optimizer = optax.adamw(learning_rate=1e-4)
    opt_state = optimizer.init(params)

    return params, optimizer, opt_state


@partial(jax.jit, static_argnames=["optimizer"])
def train_step(params, opt_state, token_ids, targets, optimizer):
    def loss_fn(p):
        return forward_and_loss(p, token_ids, targets)

    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss
