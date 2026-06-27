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

import math
from typing import NamedTuple

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


# ---------------------------------------------------------------------------
# Donoho-Tanner sparse-recovery phase transition (identifiability of sparse fits)
# ---------------------------------------------------------------------------
#
# The companion to svht/shrinkage on the *estimation* side: an identifiability
# diagnostic for sparse fits (SINDy coefficient matrices, sparse Langevin /
# Kramers-Moyal drift libraries).  Given a regression with ``n`` measurements,
# an ambient library of ``N`` candidate terms and a ``k``-sparse target, the
# Donoho-Tanner phase transition says whether ℓ1 (LASSO/STLSQ) can recover the
# support at all.  Computed via the statistical dimension of the ℓ1 descent cone
# (Amelunxen, Lotz, McCoy & Tropp 2014, "Living on the edge"), which coincides
# with the Donoho-Tanner *weak* threshold for noiseless Gaussian designs.
#
# Caveat for SINDy/Langevin: the theory assumes a (near-)Gaussian / rotationally
# invariant design.  Polynomial-feature libraries are strongly *correlated*, so
# this is the optimistic / information-theoretic bound — being above the curve is
# necessary, not sufficient; a correlated library needs strictly more samples.

_SQRT2 = math.sqrt(2.0)
_SQRT2PI = math.sqrt(2.0 * math.pi)
_erfc_vec = np.frompyfunc(math.erfc, 1, 1)


def _phi(t: np.ndarray) -> np.ndarray:
    """Standard-normal pdf."""
    return np.exp(-0.5 * t ** 2) / _SQRT2PI


def _Q(t: np.ndarray) -> np.ndarray:
    """Standard-normal upper tail 1 - Φ(t) = ½ erfc(t/√2) (vectorised)."""
    return 0.5 * _erfc_vec(t / _SQRT2).astype(float)


def l1_statistical_dimension(rho: float) -> float:
    """Normalised statistical dimension δ(ρ) of the ℓ1-norm descent cone at a
    point of relative sparsity ``rho = k/N`` (Amelunxen-Lotz-McCoy-Tropp 2014).

    Equals the Donoho-Tanner *weak* phase-transition threshold for noiseless ℓ1
    recovery: a ``k``-sparse vector in ``R^N`` is recoverable from ``n`` Gaussian
    measurements with high probability iff ``n / N > δ(ρ)``.  Monotone increasing
    on ``[0, 1]`` with ``δ(0)=0``, ``δ(1)=1`` and ``δ(ρ) ≥ ρ``.

        δ(ρ) = min_{τ≥0} ρ(1+τ²) + (1-ρ)·2[(1+τ²)Q(τ) - τφ(τ)]

    where ``φ`` is the standard-normal pdf and ``Q = 1-Φ`` its upper tail; the
    bracket is ``∫_τ^∞ (u-τ)² φ(u) du``.
    """
    rho = float(rho)
    if rho <= 0.0:
        return 0.0
    if rho >= 1.0:
        return 1.0
    tau = np.linspace(0.0, 12.0, 6001)
    moment = (1.0 + tau ** 2) * _Q(tau) - tau * _phi(tau)   # ∫_τ^∞ (u-τ)² φ du
    obj = rho * (1.0 + tau ** 2) + (1.0 - rho) * 2.0 * moment
    return float(np.min(obj))


def donoho_tanner_threshold(delta: float) -> float:
    """Donoho-Tanner *weak* phase-transition curve ``ρ_W(δ) = k/n`` for noiseless
    ℓ1 recovery from an undersampling fraction ``delta = n/N``.

    The largest sparsity (relative to the *measurements*) whose support ℓ1 still
    recovers w.h.p.  Obtained by inverting :func:`l1_statistical_dimension`:
    ``ρ_W(δ) = ψ⁻¹(δ) / δ``.  Increasing, with ``ρ_W(1)=1`` and ``ρ_W(δ)→0`` as
    ``δ→0``.  Canonical anchor: ``ρ_W(0.5) ≈ 0.385``.
    """
    delta = float(delta)
    if delta <= 0.0:
        return 0.0
    if delta >= 1.0:
        return 1.0
    lo, hi = 0.0, 1.0                       # ψ is increasing -> bisection inverse
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if l1_statistical_dimension(mid) < delta:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi) / delta


class DonohoTannerRegime(NamedTuple):
    """Where a sparse fit sits relative to the Donoho-Tanner weak threshold.

    Fields
    ------
    n_measurements, n_features, n_active : the regression geometry (n, N, k).
    delta : ``n/N`` undersampling fraction (DT x-axis).
    rho : ``k/n`` sparsity relative to measurements (DT y-axis).
    rho_crit : ``ρ_W(δ)`` weak threshold at this ``delta``.
    margin : ``rho_crit - rho`` — positive ⇔ identifiable, by how much.
    min_measurements : ``⌈N·δ(k/N)⌉`` — fewest Gaussian measurements that recover
        the support; the binding sample count.
    headroom : ``n / min_measurements`` — oversampling factor (>1 ⇔ identifiable).
    identifiable : ``n > N·δ(k/N)`` — the exact statistical-dimension condition.
    """

    n_measurements: int
    n_features: int
    n_active: int
    delta: float
    rho: float
    rho_crit: float
    margin: float
    min_measurements: int
    headroom: float
    identifiable: bool


def donoho_tanner_regime(n_measurements: int, n_features: int,
                         n_active: int) -> DonohoTannerRegime:
    """Classify a sparse-recovery problem against the Donoho-Tanner weak curve.

    Parameters
    ----------
    n_measurements : rows of the design (samples / time points), ``n``.
    n_features : columns of the design (library size), ``N``.
    n_active : number of nonzero coefficients in the target, ``k``.

    Returns
    -------
    DonohoTannerRegime
    """
    n, N, k = int(n_measurements), int(n_features), int(n_active)
    delta = n / N
    rho = k / n if n > 0 else float("inf")
    rho_crit = donoho_tanner_threshold(delta)
    stat = l1_statistical_dimension(k / N)
    min_meas = int(np.ceil(N * stat))
    return DonohoTannerRegime(
        n_measurements=n, n_features=N, n_active=k,
        delta=delta, rho=rho, rho_crit=rho_crit,
        margin=rho_crit - rho,
        min_measurements=min_meas,
        headroom=n / min_meas if min_meas > 0 else float("inf"),
        identifiable=n > N * stat,
    )
