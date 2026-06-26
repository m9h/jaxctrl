# Copyright 2024 jaxctrl contributors. Apache-2.0 license.
"""Contrastive system identification (CEBRA-style).

A differentiable, JAX/Equinox re-implementation of the time-contrastive core of
CEBRA (Schneider, Lee & Mathis, *Nature* 2023).  Contrastive self-supervised
learning is itself a system-identification method — it recovers the latent
manifold and (per DynCL, ICLR 2025) the latent dynamics — so it belongs alongside
SINDy/Koopman in jaxctrl.  With an L2-normalized 2-D output the embedding lives on
a circle, so a cyclic attractor / limit cycle appears as a ring; the recovered
latent trajectory can then be handed to :class:`SINDyOptimizer` for a governing
equation, or to control-theoretic analysis.
"""

from __future__ import annotations

from typing import List, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax


class ContrastiveEncoder(eqx.Module):
    """MLP encoder x -> latent, optionally L2-normalized onto the unit
    hypersphere (a circle for ``out_dim=2``)."""

    mlp: eqx.nn.MLP
    normalize: bool = eqx.field(static=True)

    def __init__(self, in_dim: int, out_dim: int, width: int = 64, depth: int = 2,
                 *, key, normalize: bool = True):
        self.mlp = eqx.nn.MLP(
            in_dim, out_dim, width, depth, activation=jax.nn.relu, key=key
        )
        self.normalize = normalize

    def __call__(self, x: jax.Array) -> jax.Array:
        z = self.mlp(x)
        if self.normalize:
            z = z / (jnp.linalg.norm(z) + 1e-8)
        return z


def info_nce(encoder: ContrastiveEncoder, anchors, positives, temperature):
    """Time-contrastive InfoNCE loss with in-batch negatives.

    Each anchor's positive is its temporal neighbour; every other positive in the
    batch is a negative.  ``temperature`` should be small (~0.1) for L2-normalized
    embeddings so the cosine-similarity logits span a useful range.
    """
    A = jax.vmap(encoder)(anchors)        # (B, k)
    P = jax.vmap(encoder)(positives)      # (B, k)
    logits = (A @ P.T) / temperature      # (B, B); diagonal = positive pairs
    log_prob = jax.nn.log_softmax(logits, axis=1)
    B = A.shape[0]
    return -jnp.mean(log_prob[jnp.arange(B), jnp.arange(B)])


@eqx.filter_jit
def _step(encoder, opt_state, optimizer, anchors, positives, temperature):
    loss, grads = eqx.filter_value_and_grad(info_nce)(
        encoder, anchors, positives, temperature
    )
    updates, opt_state = optimizer.update(grads, opt_state, encoder)
    encoder = eqx.apply_updates(encoder, updates)
    return encoder, opt_state, loss


class CEBRA:
    """Time-contrastive CEBRA-style latent embedding.

    Parameters
    ----------
    out_dim : latent dimensionality (2 -> a circle, good for cyclic attractors).
    width, depth : encoder MLP size.
    temperature : InfoNCE temperature (small for normalized embeddings).
    learning_rate, n_steps, batch_size : Adam training schedule.
    normalize : L2-normalize the embedding onto the unit hypersphere.
    """

    def __init__(self, out_dim: int = 2, width: int = 64, depth: int = 2,
                 temperature: float = 0.05, learning_rate: float = 1e-3,
                 n_steps: int = 1000, batch_size: int = 512, normalize: bool = True,
                 key: Optional[jax.Array] = None):
        self.out_dim = out_dim
        self.width = width
        self.depth = depth
        self.temperature = temperature
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.normalize = normalize
        self._key = key if key is not None else jr.PRNGKey(0)
        self._encoder: Optional[ContrastiveEncoder] = None

    def fit(self, X: jax.Array) -> List[float]:
        """Fit on (T, D) time-series; positive pairs are consecutive time steps."""
        X = jnp.asarray(X)
        T, D = X.shape
        key, k_enc = jr.split(self._key)
        enc = ContrastiveEncoder(D, self.out_dim, self.width, self.depth,
                                 key=k_enc, normalize=self.normalize)
        optimizer = optax.adam(self.learning_rate)
        opt_state = optimizer.init(eqx.filter(enc, eqx.is_array))
        B = min(self.batch_size, T - 1)

        history = []
        for _ in range(self.n_steps):
            key, kb = jr.split(key)
            idx = jr.randint(kb, (B,), 0, T - 1)
            enc, opt_state, loss = _step(
                enc, opt_state, optimizer, X[idx], X[idx + 1], self.temperature
            )
            history.append(float(loss))
        self._encoder = enc
        return history

    def transform(self, X: jax.Array) -> jax.Array:
        """Embed (T, D) -> (T, out_dim)."""
        if self._encoder is None:
            raise RuntimeError("CEBRA not fitted. Call fit() first.")
        return jax.vmap(self._encoder)(jnp.asarray(X))

    def __repr__(self) -> str:
        return (f"CEBRA(out_dim={self.out_dim}, temperature={self.temperature}, "
                f"n_steps={self.n_steps})")
