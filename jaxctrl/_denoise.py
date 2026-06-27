# Copyright 2024 jaxctrl contributors. Apache-2.0 license.
"""Donoho optimal denoising — parameter-free rank selection and covariance
denoising for high-dimensional, noisy data.

Replaces ad-hoc rank / threshold / regularization choices (PCA dimension, DMD
rank, harmonic truncation, covariance loading) with MSE-optimal, theory-backed
ones, and denoises high-dimensional sample covariances (HMM state covariances,
the Langevin Σ and diffusion D whose inverse amplifies estimation noise).

Methods
-------
- ``svht_*`` — optimal hard threshold for singular values (Gavish & Donoho 2014;
  the 4/√3 · y_median rule for square matrices, unknown noise).
- ``optimal_shrinkage_denoise`` — optimal nonlinear singular-value shrinkage
  (Gavish & Donoho 2017), lower MSE than hard thresholding.
- ``shrink_covariance`` — spiked-covariance eigenvalue shrinkage (Donoho, Gavish
  & Johnstone 2018): debiases sample-covariance eigenvalues toward the population.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np


# ---------------------------------------------------------------------------
# Marchenko-Pastur constants
# ---------------------------------------------------------------------------


def _lambda_star(beta: float) -> float:
    """Gavish-Donoho optimal-hard-threshold coefficient (known noise)."""
    beta = float(beta)
    return float(np.sqrt(2 * (beta + 1) + 8 * beta /
                         ((beta + 1) + np.sqrt(beta ** 2 + 14 * beta + 1))))


def _mp_median(beta: float) -> float:
    """Median of the Marchenko-Pastur distribution with ratio ``beta`` in (0,1].

    Uses a sqrt-spaced grid (dense near 0, where the density diverges as 1/√t for
    beta=1) and a normalized trapezoidal CDF."""
    beta = float(beta)
    lo, hi = (1 - np.sqrt(beta)) ** 2, (1 + np.sqrt(beta)) ** 2
    lo_eff = max(lo, hi * 1e-9)
    u = np.linspace(np.sqrt(lo_eff), np.sqrt(hi), 1_000_000)
    t = u ** 2
    dens = np.sqrt(np.clip((hi - t) * (t - lo), 0.0, None)) / (2 * np.pi * beta * t)
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (dens[1:] + dens[:-1]) * np.diff(t))])
    cdf /= cdf[-1]
    return float(t[np.searchsorted(cdf, 0.5)])


def svht_coefficient(beta: float, sigma_known: bool = False) -> float:
    """Optimal-hard-threshold coefficient: multiply by σ√n (known σ) or by the
    median singular value (unknown σ).  For a square matrix, unknown σ → 4/√3."""
    if sigma_known:
        return _lambda_star(beta)
    return _lambda_star(beta) / np.sqrt(_mp_median(beta))


# ---------------------------------------------------------------------------
# Singular-value hard thresholding (Gavish-Donoho 2014)
# ---------------------------------------------------------------------------


def _svht_threshold(s: np.ndarray, shape, sigma) -> float:
    m, n = shape
    beta = min(m, n) / max(m, n)
    if sigma is not None:
        return _lambda_star(beta) * float(sigma) * np.sqrt(max(m, n))
    return svht_coefficient(beta, False) * float(np.median(s))


def svht_rank(Y, sigma=None) -> int:
    """Optimal rank: number of singular values above the Gavish-Donoho threshold."""
    Y = np.asarray(Y)
    s = np.linalg.svd(Y, compute_uv=False)
    return int(np.sum(s > _svht_threshold(s, Y.shape, sigma)))


def svht_denoise(Y, sigma=None) -> jnp.ndarray:
    """Low-rank denoising by hard-thresholding singular values at the optimum."""
    Y = jnp.asarray(Y)
    U, s, Vt = jnp.linalg.svd(Y, full_matrices=False)
    tau = _svht_threshold(np.asarray(s), Y.shape, sigma)
    return (U * jnp.where(s > tau, s, 0.0)) @ Vt


# ---------------------------------------------------------------------------
# Optimal singular-value shrinkage (Gavish-Donoho 2017, Frobenius loss)
# ---------------------------------------------------------------------------


def _estimate_sigma(s: np.ndarray, shape) -> float:
    m, n = shape
    beta = min(m, n) / max(m, n)
    return float(np.median(s)) / (np.sqrt(max(m, n)) * np.sqrt(_mp_median(beta)))


def optimal_shrinkage_denoise(Y, sigma=None) -> jnp.ndarray:
    """Denoise via the optimal (Frobenius) singular-value shrinker — keeps weak
    components but shrinks them toward the truth, beating hard thresholding."""
    Y = jnp.asarray(Y)
    m, n = Y.shape
    beta = min(m, n) / max(m, n)
    U, s, Vt = jnp.linalg.svd(Y, full_matrices=False)
    sn = np.asarray(s)
    if sigma is None:
        sigma = _estimate_sigma(sn, Y.shape)
    scale = float(sigma) * np.sqrt(max(m, n))
    y = sn / scale
    edge = 1 + np.sqrt(beta)
    eta = np.where(y > edge,
                   np.sqrt(np.clip((y ** 2 - beta - 1) ** 2 - 4 * beta, 0.0, None)) / y,
                   0.0)
    return (U * jnp.asarray(eta * scale)) @ Vt


# ---------------------------------------------------------------------------
# Spiked-covariance eigenvalue shrinkage (Donoho-Gavish-Johnstone 2018)
# ---------------------------------------------------------------------------


def shrink_covariance(S, n_samples: int, loss: str = "operator") -> jnp.ndarray:
    """Denoise a p×p sample covariance from ``n_samples`` observations by
    shrinking its eigenvalues toward the population spikes (spiked model,
    γ = p/n).  ``loss``: 'operator' (η = debiased spike) or 'frobenius'."""
    S = np.asarray(S)
    p = S.shape[0]
    gamma = p / n_samples
    lam, V = np.linalg.eigh(S)
    v = float(np.median(lam)) / _mp_median(gamma)        # noise variance scale
    lt = lam / v
    edge = (1 + np.sqrt(gamma)) ** 2
    eta = np.ones_like(lt)
    above = lt > edge
    la = lt[above]
    ell = (la + 1 - gamma + np.sqrt(np.clip((la + 1 - gamma) ** 2 - 4 * la, 0.0, None))) / 2
    if loss == "frobenius":
        c2 = (1 - gamma / (ell - 1) ** 2) / (1 + gamma / (ell - 1))
        eta[above] = ell * c2 + (1 - c2)
    else:
        eta[above] = ell
    return jnp.asarray((V * (eta * v)) @ V.T)
