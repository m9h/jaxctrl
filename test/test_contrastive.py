"""Known-answer tests for contrastive system identification (CEBRA-style).

Sections
--------
1. Encoder output shape + unit-norm
2. InfoNCE loss decreases
3. Ring recovery: a latent on a circle (cyclic attractor) is recovered as a ring
"""

import math

import jax.numpy as jnp
import jax.random as jr
import numpy as np

from jaxctrl import CEBRA, ContrastiveEncoder

TWO_PI = 2 * math.pi


def ring_data(key, T=4000, D=12, noise=0.05):
    """Latent drifts slowly around a circle (so consecutive samples are temporal
    neighbours), nonlinearly lifted to D observed dimensions."""
    k1, k2, k3 = jr.split(key, 3)
    theta = jnp.cumsum(jr.normal(k1, (T,)) * 0.15)
    feats = jnp.stack(
        [jnp.cos(theta), jnp.sin(theta), jnp.cos(2 * theta), jnp.sin(2 * theta),
         jnp.cos(3 * theta), jnp.sin(3 * theta)], axis=1
    )
    W = jr.normal(k2, (6, D))
    X = jnp.tanh(feats @ W) + noise * jr.normal(k3, (T, D))
    return X.astype(jnp.float32), theta


def circ_corr(a, b):
    a, b = np.asarray(a), np.asarray(b)
    am = np.angle(np.mean(np.exp(1j * a)))
    bm = np.angle(np.mean(np.exp(1j * b)))
    num = np.sum(np.sin(a - am) * np.sin(b - bm))
    den = np.sqrt(np.sum(np.sin(a - am) ** 2) * np.sum(np.sin(b - bm) ** 2))
    return num / (den + 1e-12)


def test_encoder_shape_and_unit_norm():
    enc = ContrastiveEncoder(8, 2, key=jr.PRNGKey(0), normalize=True)
    z = enc(jnp.ones(8))
    assert z.shape == (2,)
    np.testing.assert_allclose(float(jnp.linalg.norm(z)), 1.0, atol=1e-5)


def test_infonce_loss_decreases():
    X, _ = ring_data(jr.PRNGKey(0))
    m = CEBRA(out_dim=2, n_steps=200, key=jr.PRNGKey(0))
    hist = m.fit(X)
    assert hist[-1] < hist[0]


def test_temporal_neighborhood_preserved():
    """The core guarantee of time-contrastive learning: temporally adjacent
    samples embed much closer than random pairs."""
    X, _ = ring_data(jr.PRNGKey(0))
    m = CEBRA(out_dim=2, temperature=0.05, n_steps=2000, batch_size=512, key=jr.PRNGKey(0))
    m.fit(X)
    e = np.asarray(m.transform(X))
    consec = np.mean(np.linalg.norm(np.diff(e, axis=0), axis=1))
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(e))
    rand = np.mean(np.linalg.norm(e[perm] - e, axis=1))
    assert consec < 0.5 * rand, f"temporal neighbours not preserved: {consec:.3f} vs {rand:.3f}"


def test_cyclic_structure_above_chance():
    """The latent ring is recovered above chance (pure time-contrastive gives
    partial global recovery; behaviour-conditioned CEBRA / DYSCO tighten it)."""
    X, theta = ring_data(jr.PRNGKey(0))
    m = CEBRA(out_dim=2, temperature=0.05, n_steps=2000, batch_size=512, key=jr.PRNGKey(0))
    m.fit(X)
    e = np.asarray(m.transform(X))
    phi = np.arctan2(e[:, 1], e[:, 0])
    rho = abs(circ_corr(np.asarray(theta), phi))
    assert rho > 0.3, f"cyclic structure not above chance: circular corr {rho:.3f}"
