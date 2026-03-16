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
