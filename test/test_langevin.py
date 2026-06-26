"""Known-answer tests for Langevin/Fokker-Planck system identification.

Sections
--------
1. Recover drift A and diffusion D from a simulated Ornstein-Uhlenbeck process
2. Reversible (symmetric-drift) system -> entropy production ~ 0 (detailed balance)
3. Rotational (antisymmetric-drift) system -> entropy production > 0, and the
   SOLENOIDAL part carries the rotation frequency omega (which a deterministic
   drift fit would miss)
4. Entropy production is non-negative
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from jaxctrl import (
    fit_linear_langevin,
    langevin_entropy_production,
    langevin_gradient_part,
    langevin_solenoidal_part,
    langevin_solenoidal_frequency,
    transition_flux,
    discrete_entropy_production,
)

TWO_PI = 2 * np.pi


def simulate_ou(A, D, dt, T, key):
    """Euler-Maruyama OU: dz = A z dt + sqrt(2 D dt) N(0,1)."""
    A = jnp.asarray(A)
    k = A.shape[0]
    L = jnp.linalg.cholesky(2 * jnp.asarray(D) * dt)
    keys = jr.split(key, T)

    def step(z, kk):
        z = z + z @ A.T * dt + jr.normal(kk, (k,)) @ L.T
        return z, z

    _, traj = jax.lax.scan(step, jnp.zeros(k), keys)
    return traj


def test_recovers_drift_and_diffusion():
    A_true = jnp.array([[-1.0, -0.5], [0.5, -1.0]])
    D_true = 0.3 * jnp.eye(2)
    dt = 0.01
    Z = simulate_ou(A_true, D_true, dt, 200000, jr.PRNGKey(0))
    m = fit_linear_langevin(Z, dt)
    np.testing.assert_allclose(np.asarray(m.A), np.asarray(A_true), atol=0.15)
    np.testing.assert_allclose(np.asarray(m.D), np.asarray(D_true), atol=0.08)


def test_reversible_zero_entropy_production():
    A = jnp.diag(jnp.array([-1.0, -2.0]))   # symmetric drift -> detailed balance
    D = 0.3 * jnp.eye(2)
    Z = simulate_ou(A, D, 0.01, 200000, jr.PRNGKey(1))
    m = fit_linear_langevin(Z, 0.01)
    eps = float(langevin_entropy_production(m))
    assert eps < 0.2, f"reversible system should have ~0 entropy production, got {eps:.3f}"


def test_rotational_positive_entropy_and_frequency():
    omega = 2.0
    A = jnp.array([[-1.0, -omega], [omega, -1.0]])  # decay + rotation
    D = 0.3 * jnp.eye(2)
    Z = simulate_ou(A, D, 0.01, 300000, jr.PRNGKey(2))
    m = fit_linear_langevin(Z, 0.01)
    eps = float(langevin_entropy_production(m))
    freq = float(langevin_solenoidal_frequency(m))
    assert eps > 0.5, f"rotational system should break detailed balance, eps={eps:.3f}"
    np.testing.assert_allclose(freq, omega / TWO_PI, atol=0.1)


def test_entropy_production_nonnegative():
    A = jnp.array([[-1.0, -1.5], [0.3, -2.0]])
    D = jnp.array([[0.4, 0.05], [0.05, 0.3]])
    Z = simulate_ou(A, D, 0.01, 100000, jr.PRNGKey(3))
    m = fit_linear_langevin(Z, 0.01)
    assert float(langevin_entropy_production(m)) > -1e-6


def test_discrete_directed_cycle_breaks_detailed_balance():
    # a directed 0->1->2->0 cycle with a little noise: strong irreversibility
    rng = np.random.default_rng(0)
    seq = []
    s = 0
    for _ in range(20000):
        s = (s + 1) % 3 if rng.random() > 0.1 else rng.integers(3)
        seq.append(s)
    seq = jnp.asarray(seq)
    eps = float(discrete_entropy_production(seq, 3))
    F = np.asarray(transition_flux(seq, 3))
    assert eps > 0.1, f"directed cycle should break detailed balance, eps={eps:.3f}"
    # net flux circulates: F_01, F_12, F_20 same sign (one rotational direction)
    assert np.sign(F[0, 1]) == np.sign(F[1, 2]) == np.sign(F[2, 0])


def test_discrete_reversible_zero_entropy():
    # symmetric random walk on a ring: detailed balance -> ~0
    rng = np.random.default_rng(1)
    s = 0
    seq = []
    for _ in range(40000):
        s = (s + rng.choice([-1, 1])) % 6
        seq.append(s)
    eps = float(discrete_entropy_production(jnp.asarray(seq), 6))
    assert eps < 0.02, f"symmetric walk should be ~reversible, eps={eps:.3f}"


def test_helmholtz_split_reconstructs_drift():
    A = jnp.array([[-1.0, -2.0], [2.0, -1.0]])
    D = 0.3 * jnp.eye(2)
    Z = simulate_ou(A, D, 0.01, 100000, jr.PRNGKey(4))
    m = fit_linear_langevin(Z, 0.01)
    recon = np.asarray(langevin_gradient_part(m) + langevin_solenoidal_part(m))
    np.testing.assert_allclose(recon, np.asarray(m.A), atol=1e-5)
