"""Controllability, observability, stabilisability, and detectability tests.

This module implements the classical structural tests for linear time-invariant
systems  dx/dt = Ax + Bu,  y = Cx:

- **Controllability** (Kalman rank condition): the controllability matrix
  [B, AB, ..., A^{n-1}B] has full row rank.

- **Observability** (dual Kalman rank condition): the observability matrix
  [C; CA; ...; CA^{n-1}] has full column rank.

- **Stabilisability** (PBH test): rank([sI - A, B]) = n for every eigenvalue s
  of A with Re(s) >= 0.  Equivalently, the uncontrollable modes are all stable.

- **Detectability** (dual PBH test): rank([sI - A; C]) = n for every unstable
  eigenvalue s.

Also provides ``minimum_energy``, the minimum control energy to steer the state
between two points in finite time.

All functions are JIT-compatible and operate on JAX arrays.
"""

from __future__ import annotations

from typing import Optional, Union

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Scalar

from jaxctrl._gramian import (
    controllability_gramian,
    controllability_matrix,
    observability_matrix,
)

Bool = Union[jnp.bool_, bool]

# Numerical tolerance for rank decisions
_DEFAULT_TOL_FACTOR = 1e-10


# ======================================================================
# Controllability
# ======================================================================

def is_controllable(
    A: Float[Array, "n n"],
    B: Float[Array, "n m"],
    tol: Optional[float] = None,
) -> Bool:
    """Check controllability via the Kalman rank condition.

    The pair (A, B) is controllable iff the controllability matrix
    [B, AB, A^2 B, ..., A^{n-1} B] has rank n.

    Parameters
    ----------
    A : (n, n) array
        System matrix.
    B : (n, m) array
        Input matrix.
    tol : float or None
        Tolerance for singular value cutoff.  If ``None``, uses
        ``n * max(sigma) * eps`` where sigma are the singular values.

    Returns
    -------
    controllable : bool scalar
    """
    n = A.shape[0]
    C_mat = controllability_matrix(A, B)
    return _has_full_row_rank(C_mat, n, tol)


# ======================================================================
# Observability
# ======================================================================

def is_observable(
    A: Float[Array, "n n"],
    C: Float[Array, "p n"],
    tol: Optional[float] = None,
) -> Bool:
    """Check observability via the Kalman rank condition.

    The pair (A, C) is observable iff the observability matrix
    [C; CA; ...; CA^{n-1}] has rank n.

    Parameters
    ----------
    A : (n, n) array
        System matrix.
    C : (p, n) array
        Output matrix.
    tol : float or None
        Tolerance for singular value cutoff.

    Returns
    -------
    observable : bool scalar
    """
    n = A.shape[0]
    O_mat = observability_matrix(A, C)
    return _has_full_column_rank(O_mat, n, tol)


# ======================================================================
# Stabilisability (PBH test)
# ======================================================================

def is_stabilizable(
    A: Float[Array, "n n"],
    B: Float[Array, "n m"],
    tol: Optional[float] = None,
) -> Bool:
    """Check stabilisability via the Popov-Belevitch-Hautus (PBH) test.

    The pair (A, B) is stabilisable iff for every eigenvalue s of A with
    Re(s) >= 0, the matrix [sI - A, B] has full row rank n.

    Equivalently, all uncontrollable modes are stable.

    Parameters
    ----------
    A : (n, n) array
    B : (n, m) array
    tol : float or None

    Returns
    -------
    stabilizable : bool scalar
    """
    n = A.shape[0]
    eigvals = jnp.linalg.eigvals(A)

    # Check unstable eigenvalues (Re >= 0)
    # For JIT compatibility, we check ALL eigenvalues and mask the stable ones.
    unstable_mask = jnp.real(eigvals) >= -_DEFAULT_TOL_FACTOR

    # Promote to complex dtype for eigenvalue arithmetic
    complex_dtype = jnp.result_type(A.dtype, jnp.complex64)
    A_c = A.astype(complex_dtype)
    B_c = B.astype(complex_dtype)

    def check_eigval(s):
        """Return True if rank([sI - A, B]) = n."""
        sI_A = s * jnp.eye(n, dtype=complex_dtype) - A_c
        M = jnp.concatenate([sI_A, B_c], axis=1)
        sv = jnp.linalg.svd(M, compute_uv=False)
        if tol is not None:
            threshold = tol
        else:
            threshold = n * jnp.max(jnp.abs(sv)) * jnp.finfo(jnp.float64).eps
        rank = jnp.sum(sv > threshold)
        return rank >= n

    # Vectorise over eigenvalues
    rank_checks = jax.vmap(check_eigval)(eigvals)

    # For stable eigenvalues the check is irrelevant — force True
    all_pass = jnp.where(unstable_mask, rank_checks, True)
    return jnp.all(all_pass)


# ======================================================================
# Detectability (dual PBH test)
# ======================================================================

def is_detectable(
    A: Float[Array, "n n"],
    C: Float[Array, "p n"],
    tol: Optional[float] = None,
) -> Bool:
    """Check detectability via the dual PBH test.

    The pair (A, C) is detectable iff (A^T, C^T) is stabilisable.

    Parameters
    ----------
    A : (n, n) array
    C : (p, n) array
    tol : float or None

    Returns
    -------
    detectable : bool scalar
    """
    return is_stabilizable(A.T, C.T, tol)


# ======================================================================
# Minimum control energy
# ======================================================================

def minimum_energy(
    A: Float[Array, "n n"],
    B: Float[Array, "n m"],
    x0: Float[Array, "n"],
    xf: Float[Array, "n"],
    T: Float[Scalar, ""],
    num_steps: int = 100,
) -> Float[Scalar, ""]:
    """Compute minimum control energy to steer state from *x0* to *xf* in time *T*.

    The minimum energy is::

        E = (x_f - e^{AT} x_0)^T  W_c(T)^{-1}  (x_f - e^{AT} x_0)

    where W_c(T) is the finite-horizon controllability Gramian.

    This is differentiable w.r.t. all inputs (A, B, x0, xf, T).

    Parameters
    ----------
    A : (n, n) array
        System matrix.
    B : (n, m) array
        Input matrix.
    x0 : (n,) array
        Initial state.
    xf : (n,) array
        Target state.
    T : scalar
        Time horizon (must be positive).
    num_steps : int
        Quadrature points for the Gramian computation.

    Returns
    -------
    E : scalar
        Minimum control energy.
    """
    eAT = jax.scipy.linalg.expm(A * T)
    W_c = controllability_gramian(A, B, T=T, num_steps=num_steps)

    delta = xf - eAT @ x0
    # E = delta^T W_c^{-1} delta
    # Solve W_c z = delta instead of computing inverse for numerical stability
    z = jnp.linalg.solve(W_c, delta)
    E = delta @ z
    return E


# ======================================================================
# Internal helpers
# ======================================================================

def _has_full_row_rank(
    M: Float[Array, "n k"],
    n: int,
    tol: Optional[float] = None,
) -> Bool:
    """Check if M has row rank >= n using SVD."""
    sv = jnp.linalg.svd(M, compute_uv=False)
    if tol is not None:
        threshold = tol
    else:
        threshold = n * jnp.max(sv) * jnp.finfo(sv.dtype).eps
    rank = jnp.sum(sv > threshold)
    return rank >= n


def _has_full_column_rank(
    M: Float[Array, "k n"],
    n: int,
    tol: Optional[float] = None,
) -> Bool:
    """Check if M has column rank >= n using SVD."""
    return _has_full_row_rank(M.T, n, tol)
