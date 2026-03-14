"""Controllability and observability Gramians for linear time-invariant systems.

This module computes the controllability and observability Gramians of a
continuous-time LTI system  dx/dt = Ax + Bu,  y = Cx.

**Controllability Gramian** (infinite horizon)::

    W_c = integral_0^inf  exp(A t) B B^T exp(A^T t) dt

which is the unique PSD solution of the Lyapunov equation::

    A W_c + W_c A^T + B B^T = 0

when A is Hurwitz.

**Observability Gramian** (infinite horizon)::

    W_o = integral_0^inf  exp(A^T t) C^T C exp(A t) dt

which solves::

    A^T W_o + W_o A + C^T C = 0

For finite-horizon Gramians, numerical quadrature with matrix exponentials is
used.

All functions are JIT-compatible and differentiable via JAX's autodiff, relying
on the differentiable Lyapunov solvers in :mod:`jaxctrl._lyapunov`.
"""

from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, Float, Scalar

from jaxctrl._lyapunov import solve_continuous_lyapunov


# ======================================================================
# Controllability Gramian
# ======================================================================

def controllability_gramian(
    A: Float[Array, "n n"],
    B: Float[Array, "n m"],
    T: Optional[Float[Scalar, ""]] = None,
    num_steps: int = 100,
) -> Float[Array, "n n"]:
    """Compute the controllability Gramian.

    Parameters
    ----------
    A : (n, n) array
        System matrix.  Must be Hurwitz for the infinite-horizon case.
    B : (n, m) array
        Input matrix.
    T : scalar or None
        Time horizon.  If ``None``, computes the infinite-horizon Gramian by
        solving the Lyapunov equation ``A W + W A^T + B B^T = 0``.
        If a positive scalar, computes the finite-horizon Gramian via
        trapezoidal quadrature.
    num_steps : int
        Number of quadrature points for the finite-horizon case (default 100).

    Returns
    -------
    W_c : (n, n) array
        Symmetric PSD controllability Gramian.
    """
    if T is None:
        # Infinite horizon: solve A W + W A^T + B B^T = 0
        Q = B @ B.T
        W_c = solve_continuous_lyapunov(A, Q)
        return W_c
    else:
        return _finite_horizon_controllability_gramian(A, B, T, num_steps)


def _finite_horizon_controllability_gramian(
    A: Float[Array, "n n"],
    B: Float[Array, "n m"],
    T: Float[Scalar, ""],
    num_steps: int,
) -> Float[Array, "n n"]:
    """Finite-horizon controllability Gramian via trapezoidal quadrature.

    W_c(T) = integral_0^T  exp(A t) B B^T exp(A^T t) dt

    Approximated with the trapezoidal rule over *num_steps* equally spaced
    points in [0, T].
    """
    n = A.shape[0]
    dt = T / num_steps
    BBT = B @ B.T

    # Endpoints with half weight
    eA0 = jnp.eye(n, dtype=A.dtype)
    f0 = eA0 @ BBT @ eA0.T
    eAT = jax.scipy.linalg.expm(A * T)
    fT = eAT @ BBT @ eAT.T
    W = 0.5 * (f0 + fT)

    # Interior points with full weight
    def body_fn(carry, t):
        eAt = jax.scipy.linalg.expm(A * t)
        val = eAt @ BBT @ eAt.T
        return carry + val, None

    interior_ts = jnp.linspace(dt, T - dt, num_steps - 1)
    W, _ = lax.scan(body_fn, W, interior_ts)

    return W * dt


# ======================================================================
# Observability Gramian
# ======================================================================

def observability_gramian(
    A: Float[Array, "n n"],
    C: Float[Array, "p n"],
    T: Optional[Float[Scalar, ""]] = None,
    num_steps: int = 100,
) -> Float[Array, "n n"]:
    """Compute the observability Gramian.

    Parameters
    ----------
    A : (n, n) array
        System matrix.  Must be Hurwitz for the infinite-horizon case.
    C : (p, n) array
        Output matrix.
    T : scalar or None
        Time horizon.  If ``None``, computes the infinite-horizon Gramian by
        solving ``A^T W + W A + C^T C = 0``.
    num_steps : int
        Number of quadrature points for the finite-horizon case.

    Returns
    -------
    W_o : (n, n) array
        Symmetric PSD observability Gramian.
    """
    if T is None:
        # Infinite horizon: solve A^T W + W A + C^T C = 0
        Q = C.T @ C
        W_o = solve_continuous_lyapunov(A.T, Q)
        return W_o
    else:
        # Duality: W_o(A, C, T) = W_c(A^T, C^T, T)
        return _finite_horizon_controllability_gramian(A.T, C.T, T, num_steps)


# ======================================================================
# Controllability and Observability matrices (Kalman)
# ======================================================================

def controllability_matrix(
    A: Float[Array, "n n"],
    B: Float[Array, "n m"],
) -> Float[Array, "n nm"]:
    """Compute the controllability matrix [B, AB, A^2 B, ..., A^{n-1} B].

    Parameters
    ----------
    A : (n, n) array
        System matrix.
    B : (n, m) array
        Input matrix.

    Returns
    -------
    C_mat : (n, n*m) array
        Controllability matrix.  The system is controllable iff this matrix
        has rank n.
    """
    n = A.shape[0]

    def scan_fn(carry, _):
        col = carry
        next_col = A @ col
        return next_col, col

    # carry starts as B, each step multiplies by A
    # We collect n terms: B, AB, A^2 B, ..., A^{n-1} B
    _, cols = lax.scan(scan_fn, B, None, length=n)
    # cols has shape (n, n, m) — stack along last axis
    # cols[i] = A^i B, shape (n, m)
    C_mat = jnp.concatenate(cols, axis=-1)  # (n, n*m)
    return C_mat


def observability_matrix(
    A: Float[Array, "n n"],
    C: Float[Array, "p n"],
) -> Float[Array, "np n"]:
    """Compute the observability matrix [C; CA; CA^2; ...; CA^{n-1}].

    Parameters
    ----------
    A : (n, n) array
        System matrix.
    C : (p, n) array
        Output matrix.

    Returns
    -------
    O_mat : (n*p, n) array
        Observability matrix.  The system is observable iff this matrix has
        rank n.
    """
    n = A.shape[0]

    def scan_fn(carry, _):
        row = carry
        next_row = row @ A
        return next_row, row

    _, rows = lax.scan(scan_fn, C, None, length=n)
    # rows has shape (n, p, n) — rows[i] = C A^i, shape (p, n)
    O_mat = jnp.concatenate(rows, axis=-2)  # (n*p, n)
    return O_mat
