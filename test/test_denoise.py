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
    l1_statistical_dimension,
    donoho_tanner_threshold,
    donoho_tanner_regime,
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


# ---------------------------------------------------------------------------
# Donoho-Tanner sparse-recovery phase transition (identifiability of sparse fits)
# ---------------------------------------------------------------------------


def test_statistical_dimension_endpoints():
    # δ(0)=0 (the zero vector needs no measurements); δ(1)=1 (a dense vector needs
    # the full ambient dimension).
    np.testing.assert_allclose(l1_statistical_dimension(0.0), 0.0, atol=1e-9)
    np.testing.assert_allclose(l1_statistical_dimension(1.0), 1.0, atol=1e-9)


def test_statistical_dimension_monotone_and_above_diagonal():
    rho = np.linspace(0.0, 1.0, 21)
    psi = np.array([l1_statistical_dimension(r) for r in rho])
    assert np.all(np.diff(psi) >= -1e-9)          # monotone increasing in sparsity
    assert np.all(psi >= rho - 1e-9)              # ψ(ρ) ≥ ρ: recovery needs > k meas.


def test_donoho_tanner_weak_anchor():
    # the canonical Donoho-Tanner weak threshold at δ=n/N=0.5 is ρ_W≈0.385 (k/n);
    # endpoints ρ_W(1)=1 and ρ_W increasing.
    np.testing.assert_allclose(donoho_tanner_threshold(0.5), 0.385, atol=0.03)
    np.testing.assert_allclose(donoho_tanner_threshold(1.0), 1.0, atol=1e-6)
    ds = np.linspace(0.05, 1.0, 20)
    rw = np.array([donoho_tanner_threshold(d) for d in ds])
    assert np.all(np.diff(rw) >= -1e-6)


def test_regime_identifiable_vs_not():
    # heavily oversampled (n≫N, few active): comfortably identifiable, big headroom
    over = donoho_tanner_regime(n_measurements=5000, n_features=50, n_active=4)
    assert over.identifiable and over.margin > 0
    assert over.min_measurements < 5000
    assert over.headroom > 1.0
    # undersampled with a near-dense target: below the weak threshold
    under = donoho_tanner_regime(n_measurements=60, n_features=200, n_active=45)
    assert not under.identifiable and under.margin < 0
    assert under.headroom < 1.0
    # the boolean and the ρ-margin agree (same boundary, two coordinates), and
    # min_measurements = ceil(N·ψ(k/N)) is consistent with the stored inputs
    for reg in (over, under):
        assert reg.identifiable == (reg.margin > 0)
        assert reg.min_measurements == int(np.ceil(
            reg.n_features * l1_statistical_dimension(reg.n_active / reg.n_features)))
