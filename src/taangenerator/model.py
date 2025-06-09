from flax.struct import dataclass
import jax
import jax.numpy as jnp
import optax


@dataclass
class TransformerParams:
    W_q: jnp.ndarray
    W_k: jnp.ndarray
    W_v: jnp.ndarray
    W_o: jnp.ndarray
    W1: jnp.ndarray
    b1: jnp.ndarray
    W2: jnp.ndarray
    b2: jnp.ndarray
    gamma1: jnp.ndarray
    beta1: jnp.ndarray
    gamma2: jnp.ndarray
    beta2: jnp.ndarray

    def tree_flatten(self):
        children = (
            self.W_q,
            self.W_k,
            self.W_v,
            self.W_o,
            self.W1,
            self.b1,
            self.W2,
            self.b2,
            self.gamma1,
            self.beta1,
            self.gamma2,
            self.beta2,
        )
        aux = None
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)


@dataclass
class StackedTransformerParams:
    blocks: tuple[TransformerParams, ...]

    def tree_flatten(self):
        return (self.blocks,), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)


@dataclass
class ModelParams:
    embedding: jax.Array
    transformer: StackedTransformerParams
    W_out: jax.Array  # final projection to vocab


def layernorm(x, gamma, beta, eps=1e-5):
    mean = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    norm = (x - mean) / jnp.sqrt(var + eps)
    return gamma * norm + beta


def attention(x, params):
    Q = x @ params.W_q
    K = x @ params.W_k
    V = x @ params.W_v

    scale = jnp.sqrt(x.shape[-1])
    # scores = Q @ K.T / scale
    scores = jnp.einsum("bij,bkj->bik", Q, K) / scale
    mask = jnp.triu(jnp.ones_like(scores), 1) * -1e9
    scores += mask
    weights = jax.nn.softmax(scores, axis=-1)
    attn = weights @ V
    return attn @ params.W_o


def feedforward(x, params):
    h = jax.nn.relu(x @ params.W1 + params.b1)
    return h @ params.W2 + params.b2


def transformer_block(x, params):
    attn_out = attention(x, params)
    x1 = layernorm(x + attn_out, params.gamma1, params.beta1)
    ff_out = feedforward(x1, params)
    x2 = layernorm(x1 + ff_out, params.gamma2, params.beta2)
    return x2


def init_transformer_params(key, d_model):
    k1, k2, k3 = jax.random.split(key, 3)

    def norm_init(shape):
        return jax.random.normal(k1, shape) / jnp.sqrt(d_model)

    def zero_init(shape):
        return jnp.zeros(shape)

    return TransformerParams(
        W_q=norm_init((d_model, d_model)),
        W_k=norm_init((d_model, d_model)),
        W_v=norm_init((d_model, d_model)),
        W_o=norm_init((d_model, d_model)),
        W1=norm_init((d_model, 4 * d_model)),
        b1=zero_init((4 * d_model,)),
        W2=norm_init((4 * d_model, d_model)),
        b2=zero_init((d_model,)),
        gamma1=jnp.ones((d_model,)),
        beta1=jnp.zeros((d_model,)),
        gamma2=jnp.ones((d_model,)),
        beta2=jnp.zeros((d_model,)),
    )


def init_embedding(key, vocab_size, d_model):
    return jax.random.normal(key, (vocab_size, d_model)) * 0.01


def embedding_fn(embedding_params, token_ids):
    return embedding_params[token_ids]


# --- Stacked Transformer ---
def init_stacked_transformer_params(key, d_model, n_layers):
    keys = jax.random.split(key, n_layers)
    return StackedTransformerParams(
        blocks=tuple(init_transformer_params(k, d_model) for k in keys)
    )


def transformer_block_batch(x, params: TransformerParams):
    attn_out = attention(x, params)
    x1 = layernorm(x + attn_out, params.gamma1, params.beta1)
    ff_out = feedforward(x1, params)
    x2 = layernorm(x1 + ff_out, params.gamma2, params.beta2)
    return x2


def stacked_forward(x, params: StackedTransformerParams):
    for block_params in params.blocks:
        x = transformer_block_batch(x, block_params)
    return x


# --- Loss ---
def cross_entropy_loss(logits, targets):
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    one_hot = jax.nn.one_hot(targets, logits.shape[-1])
    return -jnp.sum(one_hot * log_probs) / targets.shape[0]


def forward_and_loss(params: ModelParams, token_ids, targets):
    x = embedding_fn(params.embedding, token_ids)  # (B, T, D)
    x = stacked_forward(x, params.transformer)
    logits = x @ params.W_out.T  # (B, T, vocab)
    logits = logits.reshape(-1, logits.shape[-1])
    targets = targets.reshape(-1)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()
    return loss
