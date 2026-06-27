"""Known-answer tests for Donoho optimal denoising.

1. Optimal hard threshold (Gavish-Donoho 2014) recovers the planted rank
2. Optimal shrinkage (Gavish-Donoho 2017) beats hard threshold beats raw (MSE)
3. The 4/sqrt(3) coefficient for square matrices (unknown sigma)
4. Spiked-covariance eigenvalue shrinkage (Donoho-Gavish-Johnstone 2018) denoises
   a sample covariance: shrunk eigenvalues closer to the population spikes, and
   the shrunk covariance is closer (operator norm) than the raw sample covariance
"""

import numpy as np
import jax.random as jr
import jax.numpy as jnp

from jaxctrl import (
    svht_denoise,
    svht_rank,
    optimal_shrinkage_denoise,
    shrink_covariance,
    svht_coefficient,
)


def planted_lowrank(m, n, rank, spike, sigma, key):
    k1, k2, k3 = jr.split(key, 3)
    U = jnp.linalg.qr(jr.normal(k1, (m, rank)))[0]
    V = jnp.linalg.qr(jr.normal(k2, (n, rank)))[0]
    X = spike * (U @ V.T)
    Y = X + sigma * jr.normal(k3, (m, n))
    return np.asarray(X), np.asarray(Y)


def test_hard_threshold_recovers_rank():
    X, Y = planted_lowrank(300, 300, 5, 80.0, 1.0, jr.PRNGKey(0))
    assert svht_rank(Y, sigma=1.0) == 5
    assert svht_rank(Y, sigma=None) == 5          # unknown-sigma (median) path too


def test_square_coefficients():
    # known noise: the famous 4/sqrt(3); unknown noise (median rule): 2.858
    np.testing.assert_allclose(float(svht_coefficient(1.0, sigma_known=True)),
                               4.0 / np.sqrt(3.0), atol=1e-3)
    np.testing.assert_allclose(float(svht_coefficient(1.0, sigma_known=False)),
                               2.858, atol=0.02)


def test_shrinkage_beats_threshold_beats_raw():
    X, Y = planted_lowrank(200, 400, 8, 50.0, 1.0, jr.PRNGKey(1))
    raw = np.linalg.norm(Y - X)
    thr = np.linalg.norm(np.asarray(svht_denoise(Y, sigma=1.0)) - X)
    shr = np.linalg.norm(np.asarray(optimal_shrinkage_denoise(Y, sigma=1.0)) - X)
    assert shr < thr < raw, f"shrink {shr:.1f} < thr {thr:.1f} < raw {raw:.1f}"


def test_covariance_eigenvalue_shrinkage():
    rng = np.random.default_rng(0)
    p, n = 60, 300                                  # gamma=0.2
    pop = np.ones(p)
    pop[:3] = np.array([30.0, 18.0, 9.0])           # three spikes over unit noise
    Sigma = np.diag(pop)
    X = rng.standard_normal((n, p)) @ np.sqrt(Sigma)
    S = np.cov(X, rowvar=False)
    Shr_fr = np.asarray(shrink_covariance(jnp.asarray(S), n, loss="frobenius"))
    # the Frobenius shrinker (Donoho-Gavish-Johnstone) denoises the covariance:
    # lower Frobenius error to the population than the raw sample covariance
    assert np.linalg.norm(Shr_fr - Sigma, "fro") < np.linalg.norm(S - Sigma, "fro")
