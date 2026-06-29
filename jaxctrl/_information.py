# Copyright 2024 jaxctrl contributors. Apache-2.0 license.
"""JAX-native Gaussian multivariate transfer entropy + permutation tests.

The Gaussian conditional-MI estimator (IDTxl's ``JidtGaussianCMI`` = linear transfer
entropy = conditional Granger causality) reduces to log-determinants of covariance
sub-blocks — fully differentiable and GPU-``vmap``-able.  Multivariate conditioning
(on the rest of the network's past) removes common-driver / cascade confounds
(Novelli et al. 2019); the permutation null (circular shifts of the source) is
embarrassingly parallel, vmap-ed over hundreds of shifts.

A faster, differentiable alternative to the IDTxl/JIDT oracle for the Gaussian
(linear) regime; nonlinear TE would need a kNN (KSG) or neural (DINE/TREET) estimator.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def _as2d(x):
    x = jnp.asarray(x)
    return x[:, None] if x.ndim == 1 else x


def _logdet_cov(A, ridge):
    A = A - jnp.mean(A, axis=0, keepdims=True)
    C = (A.T @ A) / (A.shape[0] - 1) + ridge * jnp.eye(A.shape[1])
    return jnp.linalg.slogdet(C)[1]


def gaussian_cmi(X, Y, Z=None, ridge=1e-6):
    """Gaussian conditional mutual information ``I(X;Y|Z)`` (nats) from samples.

    ``X`` (T, dx), ``Y`` (T, dy), ``Z`` (T, dz) or None.  For jointly-Gaussian
    variables this is exact; otherwise it is the linear (second-order) estimate."""
    X, Y = _as2d(X), _as2d(Y)
    if Z is None:
        XY = jnp.concatenate([X, Y], axis=1)
        return 0.5 * (_logdet_cov(X, ridge) + _logdet_cov(Y, ridge) - _logdet_cov(XY, ridge))
    Z = _as2d(Z)
    XZ = jnp.concatenate([X, Z], axis=1)
    YZ = jnp.concatenate([Y, Z], axis=1)
    XYZ = jnp.concatenate([X, Y, Z], axis=1)
    return 0.5 * (_logdet_cov(XZ, ridge) + _logdet_cov(YZ, ridge)
                  - _logdet_cov(XYZ, ridge) - _logdet_cov(Z, ridge))


def _embed(x, lags, maxlag):
    """Delay embedding (T-maxlag, n_lags) of a 1-D series at the given positive lags."""
    x = jnp.asarray(x)
    T = x.shape[0]
    return jnp.stack([x[maxlag - l: T - l] for l in lags], axis=1)


def transfer_entropy(source, target, cond=None, source_lags=(1, 2),
                     target_lags=(1, 2), ridge=1e-6):
    """Transfer entropy ``TE(source→target | cond)`` (nats), Gaussian estimator.

    ``TE = I(target_future; source_past | target_past, cond_past)``.  ``cond`` is an
    optional (T, k) array of conditioning series (their pasts are included)."""
    maxlag = max(max(source_lags), max(target_lags))
    yf = jnp.asarray(target)[maxlag:][:, None]                 # target future
    sp = _embed(source, source_lags, maxlag)                   # source past
    Zpast = _embed(target, target_lags, maxlag)                # target past (always conditioned)
    if cond is not None:
        cond = _as2d(cond)
        cps = [_embed(cond[:, j], source_lags, maxlag) for j in range(cond.shape[1])]
        Zpast = jnp.concatenate([Zpast] + cps, axis=1)
    return gaussian_cmi(sp, yf, Zpast, ridge)


def mvte_matrix(data, source_lags=(1, 2), target_lags=(1, 2), ridge=1e-6):
    """Multivariate conditional TE matrix ``M[s, t] = TE(s→t | all other pasts)``.

    ``data`` (T, n).  Each directed edge is conditioned on every *other* node's past,
    removing common-driver and cascade confounds.  Diagonal is zero.  Vectorised: a
    single ``vmap`` over all ordered pairs (compiles once)."""
    data = jnp.asarray(data)
    n = data.shape[1]
    pairs = jnp.array([(s, t) for s in range(n) for t in range(n) if s != t])
    cond_idx = jnp.array([[j for j in range(n) if j != s and j != t]
                          for s, t in [(int(p[0]), int(p[1])) for p in pairs]])

    def one(pair, cidx):
        cond = data[:, cidx] if cidx.shape[0] > 0 else None
        return transfer_entropy(data[:, pair[0]], data[:, pair[1]], cond,
                                source_lags, target_lags, ridge)

    vals = jax.vmap(one)(pairs, cond_idx)
    return jnp.zeros((n, n)).at[pairs[:, 0], pairs[:, 1]].set(vals)


def te_permutation_test(source, target, cond=None, source_lags=(1, 2),
                        target_lags=(1, 2), n_perm=200, key=None, ridge=1e-6):
    """Significance of ``TE(source→target | cond)`` vs a circular-shift null.

    The source is circularly shifted by random offsets (preserving its own
    autocorrelation, destroying the source→target timing); the null TEs are computed
    by ``vmap`` over shifts.  Returns ``(observed, null, z, p)``."""
    source = jnp.asarray(source)
    T = source.shape[0]
    maxlag = max(max(source_lags), max(target_lags))
    obs = transfer_entropy(source, target, cond, source_lags, target_lags, ridge)
    if key is None:
        key = jax.random.PRNGKey(0)
    shifts = jax.random.randint(key, (n_perm,), maxlag + 1, T - maxlag - 1)
    idx0 = jnp.arange(T)

    def null_te(shift):
        rolled = source[(idx0 - shift) % T]
        return transfer_entropy(rolled, target, cond, source_lags, target_lags, ridge)

    null = jax.vmap(null_te)(shifts)
    z = (obs - jnp.mean(null)) / (jnp.std(null) + 1e-12)
    p = (jnp.sum(null >= obs) + 1) / (n_perm + 1)
    return obs, null, z, p
