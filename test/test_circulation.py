"""Known-answer tests for the Lévy-area / irreversible-circulation tools.

The antisymmetric level-2 signature (Lévy area) is the net circulation of a path;
for a linear Langevin it equals the Tomita-Tomita irreversible-circulation matrix
α* = ½(AΣ − ΣAᵀ) = A_sol·Σ (Strang 2024), the path-space twin of the solenoidal
drift. These tests pin (i) the geometry (circle → π, line → 0, reversal flips
sign), and (ii) the OU identity linking the empirical path circulation to the
Langevin drift/covariance.
"""

import numpy as np
import jax.numpy as jnp
from scipy.linalg import solve_lyapunov

from jaxctrl import (
    levy_area,
    levy_rate,
    circulation_strength,
    expected_levy_rate,
    solenoidal_circulation,
)

TWO_PI = 2 * np.pi


def test_circle_levy_area_is_pi():
    th = jnp.linspace(0, TWO_PI, 4000)
    circ = jnp.stack([jnp.cos(th), jnp.sin(th)], axis=1)        # anticlockwise
    A = np.asarray(levy_area(circ))
    np.testing.assert_allclose(A[0, 1], np.pi, atol=2e-2)       # enclosed area = π
    np.testing.assert_allclose(A, -A.T, atol=1e-6)              # antisymmetric
    # clockwise reverses the sign
    cw = jnp.stack([jnp.cos(th), -jnp.sin(th)], axis=1)
    assert float(levy_area(cw)[0, 1]) < 0


def test_straight_line_zero_area():
    t = jnp.linspace(0, 1, 1000)
    line = jnp.stack([t, 2.0 * t + 0.3], axis=1)               # no enclosed area
    np.testing.assert_allclose(float(levy_area(line)[0, 1]), 0.0, atol=1e-6)


def test_time_reversal_negates():
    rng = np.random.default_rng(0)
    path = jnp.asarray(np.cumsum(rng.standard_normal((2000, 3)), axis=0))
    A = np.asarray(levy_area(path))
    Arev = np.asarray(levy_area(path[::-1]))
    np.testing.assert_allclose(Arev, -A, atol=1e-5)


def test_solenoidal_identity():
    # A_sol·Σ == ½(AΣ − ΣAᵀ) == −expected_levy_rate(A, Σ), exactly
    rng = np.random.default_rng(1)
    r = 4
    M = rng.standard_normal((r, r))
    A = M - (np.abs(np.linalg.eigvals(M)).max() + 0.8) * np.eye(r)   # stable
    Dh = rng.standard_normal((r, r))
    D = Dh @ Dh.T / r + 0.4 * np.eye(r)
    Sig = solve_lyapunov(A, -2 * D)
    Sig = 0.5 * (Sig + Sig.T)
    A_sol = A + D @ np.linalg.inv(Sig)
    alpha = solenoidal_circulation(jnp.asarray(A_sol), jnp.asarray(Sig))
    np.testing.assert_allclose(np.asarray(alpha), 0.5 * (A @ Sig - Sig @ A.T), atol=1e-8)
    np.testing.assert_allclose(np.asarray(alpha),
                               -np.asarray(expected_levy_rate(jnp.asarray(A), jnp.asarray(Sig))),
                               atol=1e-8)


def test_ou_path_circulation_matches_drift():
    # the empirical Itô Lévy-area RATE of a simulated OU path ≈ expected_levy_rate(A,Σ)
    rng = np.random.default_rng(0)
    r = 3
    A = np.array([[-2.0, -1.0, 0.0], [1.0, -2.0, -0.5], [0.0, 0.5, -1.5]])
    Dh = rng.standard_normal((r, r))
    D = Dh @ Dh.T / r + 0.5 * np.eye(r)
    Sig = solve_lyapunov(A, -2 * D)
    Sig = 0.5 * (Sig + Sig.T)
    dt, n = 1e-3, 400_000
    Lc = np.linalg.cholesky(2 * D)
    z = np.zeros(r)
    Z = np.empty((n, r))
    sq = np.sqrt(dt)
    for t in range(n):
        z = z + A @ z * dt + Lc @ rng.standard_normal(r) * sq
        Z[t] = z
    emp = np.asarray(levy_rate(jnp.asarray(Z), dt))
    pred = np.asarray(expected_levy_rate(jnp.asarray(A), jnp.asarray(Sig)))
    # dominant off-diagonal entry matches in sign and ~magnitude (finite-sample)
    i, j = np.unravel_index(np.argmax(np.abs(pred)), pred.shape)
    np.testing.assert_allclose(emp[i, j], pred[i, j], rtol=0.15)
    assert circulation_strength(jnp.asarray(Z), dt) > 0
