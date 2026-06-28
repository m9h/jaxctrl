# Copyright 2024 jaxctrl contributors. Apache-2.0 license.
"""System identification: SINDy and DMD/Koopman.

Data-driven methods for identifying dynamical systems from time-series
observations.  The identified models produce (A, B) matrices suitable
for analysis with jaxctrl's control-theoretic tools.

References
----------
  Brunton, Proctor & Kutz (2016).  Discovering governing equations from
    data by sparse identification of nonlinear dynamical systems.  PNAS.
  Tu, Rowley, Luchtenburg, Brunton & Kutz (2014).  On dynamic mode
    decomposition: Theory and applications.  J. Comp. Dyn.
"""

from __future__ import annotations

import functools
from itertools import combinations_with_replacement
from typing import Callable, Optional, Tuple

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx


# ---------------------------------------------------------------------------
# Feature libraries
# ---------------------------------------------------------------------------


def polynomial_library(X: jax.Array, degree: int = 2) -> jax.Array:
    """Build polynomial feature library from state observations.

    Parameters
    ----------
    X : array, shape (n_samples, n_vars)
        State observations.
    degree : int
        Maximum polynomial degree (1=linear, 2=quadratic, ...).

    Returns
    -------
    Theta : array, shape (n_samples, n_features)
        Feature library.  Columns are ordered: constant, degree-1
        monomials, degree-2 monomials, etc.
    """
    n_samples, n_vars = X.shape
    terms = [jnp.ones((n_samples, 1))]

    for d in range(1, degree + 1):
        for combo in combinations_with_replacement(range(n_vars), d):
            col = jnp.ones(n_samples)
            for idx in combo:
                col = col * X[:, idx]
            terms.append(col[:, None])

    return jnp.concatenate(terms, axis=1)


def fourier_library(
    X: jax.Array,
    n_freqs: int = 3,
) -> jax.Array:
    """Build Fourier feature library (sin/cos pairs).

    Useful for systems with periodic dynamics.

    Parameters
    ----------
    X : array, shape (n_samples, n_vars)
    n_freqs : int
        Number of frequency harmonics per variable.

    Returns
    -------
    Theta : array, shape (n_samples, 1 + 2 * n_vars * n_freqs)
    """
    n_samples, n_vars = X.shape
    terms = [jnp.ones((n_samples, 1))]

    for k in range(1, n_freqs + 1):
        for j in range(n_vars):
            terms.append(jnp.cos(k * X[:, j : j + 1]))
            terms.append(jnp.sin(k * X[:, j : j + 1]))

    return jnp.concatenate(terms, axis=1)


# ---------------------------------------------------------------------------
# SINDy
# ---------------------------------------------------------------------------


class SINDyOptimizer(eqx.Module):
    """Sparse Identification of Nonlinear Dynamics (SINDy).

    Identifies a sparse dynamical system ``dX/dt = Theta(X) @ Xi`` from
    state observations *X* and derivatives *dX* using Sequential
    Thresholded Least Squares (STLSQ).

    Parameters
    ----------
    threshold : float
        Coefficient magnitude below which entries are zeroed.
    max_iter : int
        Number of STLSQ iterations (threshold then re-solve).
    """

    threshold: float = 0.1
    max_iter: int = 10

    def fit(
        self,
        X: jax.Array,
        dX: jax.Array,
        library_fn: Callable[[jax.Array], jax.Array],
    ) -> jax.Array:
        """Fit SINDy model to data.

        Parameters
        ----------
        X : array, shape (n_samples, n_vars)
            State trajectory.
        dX : array, shape (n_samples, n_vars)
            Derivative trajectory.
        library_fn : callable
            Maps X to the feature library Theta(X).

        Returns
        -------
        Xi : array, shape (n_library_features, n_vars)
            Sparse coefficient matrix.
        """
        Theta = library_fn(X)

        # Use float64 numpy for the least-squares solves to avoid
        # float32 precision issues with ill-conditioned library matrices.
        Theta_np = np.asarray(Theta, dtype=np.float64)
        dX_np = np.asarray(dX, dtype=np.float64)

        # Initial least-squares solve
        Xi_np = np.linalg.lstsq(Theta_np, dX_np, rcond=None)[0]

        # STLSQ: threshold then re-solve on the support
        for _ in range(self.max_iter):
            mask = np.abs(Xi_np) >= self.threshold

            for j in range(dX_np.shape[1]):
                support = mask[:, j]
                if not np.any(support):
                    Xi_np[:, j] = 0.0
                    continue
                # Re-solve on the support columns only
                xi_s = np.linalg.lstsq(
                    Theta_np[:, support], dX_np[:, j], rcond=None
                )[0]
                Xi_np[:, j] = 0.0
                Xi_np[support, j] = xi_s

        return jnp.array(Xi_np, dtype=Theta.dtype)

    def predict(
        self,
        X: jax.Array,
        Xi: jax.Array,
        library_fn: Callable[[jax.Array], jax.Array],
    ) -> jax.Array:
        """Predict derivatives: dX_pred = Theta(X) @ Xi."""
        return library_fn(X) @ Xi

    @staticmethod
    def linearize(
        Xi: jax.Array,
        n_vars: int,
        library_fn: Optional[Callable] = None,
        x_eq: Optional[jax.Array] = None,
    ) -> jax.Array:
        """Extract linear system matrix A from SINDy coefficients.

        For a polynomial library the linear terms occupy rows
        ``1 : n_vars + 1`` of Xi (row 0 is the constant).  At the
        origin this block *is* the Jacobian.

        For an arbitrary equilibrium, pass *x_eq* and *library_fn*
        to compute the Jacobian via JAX autodiff.

        Parameters
        ----------
        Xi : array, shape (n_library, n_vars)
        n_vars : int
            Number of state variables.
        library_fn : callable, optional
            Required when *x_eq* is not None.
        x_eq : array, shape (n_vars,), optional
            Equilibrium point.  If None, linearises at the origin
            using the polynomial coefficient block.

        Returns
        -------
        A : array, shape (n_vars, n_vars)
        """
        if x_eq is None:
            # For polynomial library: rows 1..n_vars are the linear
            # coefficients.  Transpose to get (n_vars, n_vars).
            return Xi[1 : n_vars + 1, :].T

        if library_fn is None:
            raise ValueError("library_fn required when x_eq is given")

        def f(x):
            Theta = library_fn(x[None, :])
            return (Theta @ Xi).squeeze()

        return jax.jacobian(f)(x_eq)


# ---------------------------------------------------------------------------
# DMD / Koopman
# ---------------------------------------------------------------------------


class KoopmanEstimator(eqx.Module):
    """Koopman operator estimation via Exact Dynamic Mode Decomposition.

    Given snapshot pairs (X, Y) where ``Y ≈ K @ X``, estimates the
    finite-dimensional approximation of the Koopman operator.

    Parameters
    ----------
    rank : int
        SVD truncation rank.  0 (default) keeps all singular values.
    """

    rank: int = 0

    def fit(
        self,
        X: jax.Array,
        Y: jax.Array,
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """Fit Koopman operator K such that Y ≈ K @ X.

        Parameters
        ----------
        X : array, shape (n_features, n_samples)
            Snapshots at time t.
        Y : array, shape (n_features, n_samples)
            Snapshots at time t + dt.

        Returns
        -------
        K : array, shape (n_features, n_features)
            Koopman operator approximation.
        eigenvalues : array, shape (r,)
            DMD eigenvalues.
        modes : array, shape (n_features, r)
            DMD modes (columns).
        """
        U, S, Vh = jnp.linalg.svd(X, full_matrices=False)
        V = Vh.T.conj()

        r = self.rank if self.rank > 0 else len(S)
        r = min(r, len(S))

        Ur, Sr, Vr = U[:, :r], S[:r], V[:, :r]
        Sr_inv = jnp.diag(1.0 / Sr)

        # Projected operator
        Atilde = Ur.T.conj() @ Y @ Vr @ Sr_inv

        # Eigendecomposition
        eigenvalues, W = jnp.linalg.eig(Atilde)

        # DMD modes
        modes = Y @ Vr @ Sr_inv @ W

        # Full-space operator
        K = Ur @ Atilde @ Ur.T.conj()

        return K, eigenvalues, modes

    def predict(
        self,
        x0: jax.Array,
        t: int,
        eigenvalues: jax.Array,
        modes: jax.Array,
    ) -> jax.Array:
        """Predict state at step *t* using eigendecomposition.

        ``x_t = Phi @ diag(lambda^t) @ Phi^+ @ x0``

        More efficient than matrix_power for large *t*.

        Parameters
        ----------
        x0 : array, shape (n_features,)
        t : int
            Number of time steps.
        eigenvalues : array, shape (r,)
        modes : array, shape (n_features, r)

        Returns
        -------
        x_t : array, shape (n_features,)
        """
        Phi_pinv = jnp.linalg.pinv(modes)
        b = Phi_pinv @ x0
        lambda_t = eigenvalues ** t
        return jnp.real(modes @ (lambda_t * b))

    @staticmethod
    def continuous_eigenvalues(
        eigenvalues: jax.Array,
        dt: float,
    ) -> jax.Array:
        """Convert discrete DMD eigenvalues to continuous-time.

        ``omega = log(lambda) / dt``
        """
        return jnp.log(eigenvalues + 0j) / dt

    @staticmethod
    def is_stable(eigenvalues: jax.Array) -> jax.Array:
        """Check discrete-time stability (all |lambda| < 1)."""
        return jnp.all(jnp.abs(eigenvalues) < 1.0)


# ---------------------------------------------------------------------------
# S-map / EDM (sequential locally-weighted linear maps; Sugihara 1994)
# ---------------------------------------------------------------------------


def smap_predict(
    library_X: jax.Array,
    library_Y: jax.Array,
    query_X: jax.Array,
    theta: float,
    ridge: float = 1e-6,
) -> jax.Array:
    """S-map forecast: a locally-weighted (by ``theta``) linear map per query point.

    For each query the library points are weighted ``w_i = exp(-theta·d_i/d̄)`` (d_i
    the distance to library point i, d̄ their mean) and a ridge-regularised linear
    map is fit and applied.  ``theta=0`` is a single global linear map; larger
    ``theta`` localises to a state-dependent (nonlinear) map.

    Parameters
    ----------
    library_X : (L, d) library states.
    library_Y : (L, p) library targets (e.g. the lag-ahead state).
    query_X : (M, d) query states.
    theta : locality parameter (>= 0).
    ridge : Tikhonov regularisation for the weighted normal equations.

    Returns
    -------
    (M, p) predictions.
    """
    library_X = jnp.asarray(library_X)
    library_Y = jnp.asarray(library_Y)
    query_X = jnp.asarray(query_X)
    L, d = library_X.shape
    Xa = jnp.concatenate([jnp.ones((L, 1)), library_X], axis=1)          # (L, d+1)
    dist = jnp.sqrt(jnp.maximum(
        jnp.sum((query_X[:, None, :] - library_X[None, :, :]) ** 2, axis=-1), 0.0))
    dbar = jnp.mean(dist, axis=1, keepdims=True) + 1e-12                 # (M, 1)
    W = jnp.exp(-theta * dist / dbar)                                    # (M, L)
    # per-query weighted ridge: beta = (Xaᵀ W Xa + ridge I)⁻¹ Xaᵀ W Y
    A = jnp.einsum("ml,lj,lk->mjk", W, Xa, Xa) + ridge * jnp.eye(d + 1)
    b = jnp.einsum("ml,lj,lp->mjp", W, Xa, library_Y)
    beta = jnp.linalg.solve(A, b)                                       # (M, d+1, p)
    qa = jnp.concatenate([jnp.ones((query_X.shape[0], 1)), query_X], axis=1)
    return jnp.einsum("mj,mjp->mp", qa, beta)


def smap_nonlinearity(
    Z: jax.Array,
    thetas,
    lag: int = 1,
    library_frac: float = 0.5,
    ridge: float = 1e-6,
    theiler: int = 10,
    max_pts: int = 2000,
) -> jax.Array:
    """S-map forecast skill ρ(θ) — a *positive* nonlinearity / determinism test.

    Standardises ``Z`` (T, d), predicts the ``lag``-ahead state from a library (first
    ``library_frac`` of the record) on a held-out query set (with a ``theiler`` gap),
    and returns the mean cross-correlation skill for each θ in ``thetas``.  Skill that
    *rises* with θ ⇒ state-dependent (nonlinear, deterministic) dynamics; flat/falling
    ⇒ a global linear map suffices (linear / stochastic).  Library and query are
    strided to ``max_pts`` points to bound the O(L·M) cost.
    """
    Z = jnp.asarray(Z)
    Z = (Z - jnp.mean(Z, axis=0)) / (jnp.std(Z, axis=0) + 1e-12)
    X, Y = Z[:-lag], Z[lag:]
    n = X.shape[0]
    nlib = int(n * library_frac)
    li = jnp.linspace(0, nlib - 1, min(max_pts, nlib)).astype(int)
    qi = jnp.linspace(nlib + theiler, n - 1,
                      min(max_pts, n - nlib - theiler)).astype(int)
    libX, libY, qX, qY = X[li], Y[li], X[qi], Y[qi]

    def skill(theta):
        pred = smap_predict(libX, libY, qX, float(theta), ridge)
        pc = pred - jnp.mean(pred, axis=0)
        yc = qY - jnp.mean(qY, axis=0)
        num = jnp.sum(pc * yc, axis=0)
        den = jnp.sqrt(jnp.sum(pc ** 2, axis=0) * jnp.sum(yc ** 2, axis=0)) + 1e-12
        return jnp.mean(num / den)

    return jnp.stack([skill(float(t)) for t in np.asarray(thetas)])
