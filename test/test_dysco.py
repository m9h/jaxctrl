"""Known-answer tests for DYSCO (governing equations from latent dynamics).

Sections
--------
1. Shapes (fit / transform / coefficients)
2. Loss decreases
3. Rotational recovery: from a nonlinearly-observed 2-D rotation (a limit-cycle),
   DYSCO recovers a latent linear flow whose eigenvalues are ~ +-i*omega
   (imaginary -> oscillation), up to the affine gauge.
"""

import jax.numpy as jnp
import jax.random as jr
import numpy as np

from jaxctrl import DYSCO


def rotation_data(key, T=3000, D=12, omega=1.0, dt=0.1, noise=0.02):
    """Latent rotates on a circle (ż = [[0,-w],[w,0]] z); nonlinearly lifted."""
    k1, k2 = jr.split(key)
    t = jnp.arange(T) * dt
    z = jnp.stack([jnp.cos(omega * t), jnp.sin(omega * t)], axis=1)  # (T, 2)
    feats = jnp.stack(
        [z[:, 0], z[:, 1], z[:, 0] ** 2, z[:, 1] ** 2, z[:, 0] * z[:, 1]], axis=1
    )
    W = jr.normal(k1, (5, D))
    X = jnp.tanh(feats @ W) + noise * jr.normal(k2, (T, D))
    return X.astype(jnp.float32), z


def test_shapes():
    X, _ = rotation_data(jr.PRNGKey(0))
    m = DYSCO(latent_dim=2, library_degree=1, n_steps=50, key=jr.PRNGKey(0))
    m.fit(X)
    assert m.transform(X).shape == (X.shape[0], 2)
    assert m.coefficients().shape[1] == 2
    assert m.linear_part().shape == (2, 2)


def test_loss_decreases():
    X, _ = rotation_data(jr.PRNGKey(0))
    m = DYSCO(latent_dim=2, library_degree=1, n_steps=300, key=jr.PRNGKey(0))
    hist = m.fit(X)
    assert hist[-1] < hist[0]


def test_recovers_rotational_dynamics():
    omega = 1.0
    X, _ = rotation_data(jr.PRNGKey(0), omega=omega, dt=0.1)
    m = DYSCO(latent_dim=2, library_degree=1, dt=0.1, n_steps=4000,
              batch_size=256, key=jr.PRNGKey(0))
    m.fit(X)
    A = np.asarray(m.linear_part())
    evals = np.linalg.eigvals(A)
    # gauge-invariant eigenvalues should be ~ +-i*omega: near-imaginary, |Im| ~ omega
    re = np.abs(evals.real)
    im = np.abs(evals.imag)
    assert np.max(re) < 0.4, f"eigenvalues not near-imaginary: {evals}"
    assert 0.5 < np.mean(im) < 1.6, f"oscillation frequency off: {evals} (omega={omega})"
