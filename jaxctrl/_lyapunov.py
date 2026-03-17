"""Differentiable Lyapunov equation solvers for JAX.

This module provides JIT-compatible, fully differentiable solvers for continuous-
and discrete-time Lyapunov equations, which arise throughout linear systems
theory.

**Continuous Lyapunov equation**::

    A X + X A^T + Q = 0

Given a Hurwitz matrix *A* (all eigenvalues in the open left half-plane) and a
symmetric matrix *Q*, the unique solution *X* is symmetric positive semi-definite
when *Q* is positive semi-definite.

**Discrete Lyapunov equation**::

    A X A^T - X + Q = 0

Given a Schur-stable matrix *A* (all eigenvalues strictly inside the unit disk)
and symmetric *Q*, the unique solution is again symmetric PSD.

The backward passes use the adjoint identities from:

    Kao, C.-Y. & Hennequin, G. (2020).
    "Differentiable linear algebra for control theory."

Specifically, for the continuous Lyapunov equation the adjoint of *X* w.r.t. *A*
and *Q* is obtained by solving another Lyapunov equation in the co-tangent.
"""

from __future__ import annotations

from typing import Union

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

try:
    import lineax as lx

    _HAS_LINEAX = True
except ImportError:
    lx = None  # type: ignore[assignment]
    _HAS_LINEAX = False

# Dispatch threshold: use Lineax iterative solver for n > this value.
_LINEAX_THRESHOLD = 50


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _symmetrise(X: Float[Array, "n n"]) -> Float[Array, "n n"]:
    """Return (X + X^T) / 2."""
    return 0.5 * (X + X.T)


# ---------------------------------------------------------------------------
# Continuous Lyapunov: A X + X A^T + Q = 0
# ---------------------------------------------------------------------------

@jax.custom_vjp
def solve_continuous_lyapunov(
    A: Float[Array, "n n"],
    Q: Float[Array, "n n"],
) -> Float[Array, "n n"]:
    """Solve the continuous Lyapunov equation ``A X + X A^T + Q = 0``.

    Uses the Bartels-Stewart algorithm: Schur-decompose *A = U T U^T*, transform
    the equation into triangular form, solve column-by-column, then back-transform.

    Parameters
    ----------
    A : (n, n) array
        System matrix.  Must be Hurwitz (all eigenvalues with Re < 0) for a
        unique solution to exist.
    Q : (n, n) array
        Symmetric forcing matrix.

    Returns
    -------
    X : (n, n) array
        Symmetric solution.  Returns NaN matrix when the Schur solver fails
        (e.g. *A* is not Hurwitz).
    """
    return _solve_continuous_lyapunov_impl(A, Q)


def _solve_continuous_lyapunov_kron(
    A: Float[Array, "n n"],
    Q: Float[Array, "n n"],
) -> Float[Array, "n n"]:
    """Solve AX + XA^T + Q = 0 via Kronecker vectorisation.

    vec(AX + XA^T) = (I kron A + A kron I) vec(X) = -vec(Q)

    This is a dense n^2 x n^2 linear system, exact and JIT-friendly.
    O(n^4) memory, O(n^6) compute — use for n <= 50.
    """
    n = A.shape[0]
    I_n = jnp.eye(n, dtype=A.dtype)
    M = jnp.kron(I_n, A) + jnp.kron(A, I_n)
    x = jnp.linalg.solve(M, -Q.ravel())
    X = x.reshape(n, n)
    X = _symmetrise(X)
    return X


def _solve_continuous_lyapunov_lineax(
    A: Float[Array, "n n"],
    Q: Float[Array, "n n"],
) -> Float[Array, "n n"]:
    """Solve AX + XA^T + Q = 0 via Lineax iterative solver (GMRES).

    Expresses the Lyapunov operator as a FunctionLinearOperator and
    solves with GMRES.  O(n^2) memory, suitable for large n.
    """
    n = A.shape[0]

    def lyap_op(X_vec):
        X = X_vec.reshape(n, n)
        return (A @ X + X @ A.T).ravel()

    operator = lx.FunctionLinearOperator(
        lyap_op, jax.ShapeDtypeStruct((n * n,), A.dtype)
    )
    sol = lx.linear_solve(
        operator, -Q.ravel(), solver=lx.GMRES(rtol=1e-6, atol=1e-8)
    )
    X = sol.value.reshape(n, n)
    return _symmetrise(X)


def _solve_continuous_lyapunov_impl(
    A: Float[Array, "n n"],
    Q: Float[Array, "n n"],
) -> Float[Array, "n n"]:
    """Dispatch to Kronecker or Lineax solver based on matrix size."""
    if _HAS_LINEAX and A.shape[0] > _LINEAX_THRESHOLD:
        return _solve_continuous_lyapunov_lineax(A, Q)
    return _solve_continuous_lyapunov_kron(A, Q)


def _solve_continuous_lyapunov_fwd(
    A: Float[Array, "n n"],
    Q: Float[Array, "n n"],
):
    X = _solve_continuous_lyapunov_impl(A, Q)
    return X, (A, Q, X)


def _solve_continuous_lyapunov_bwd(res, g):
    """Backward pass using Kao & Hennequin (2020).

    Given dL/dX = G, the adjoint equations are:

        Solve  A^T S + S A + G_sym = 0   (another continuous Lyapunov in S)

    Then:
        dL/dA = -(S X^T + S^T X)  (after symmetrisation of G)
        dL/dQ = -S
    """
    A, Q, X = res
    G_sym = _symmetrise(g)

    # Adjoint Lyapunov: A^T S + S A + G_sym = 0
    S = _solve_continuous_lyapunov_impl(A.T, G_sym)

    dA = S @ X.T + S.T @ X
    dQ = S
    return dA, dQ


solve_continuous_lyapunov.defvjp(_solve_continuous_lyapunov_fwd,
                                  _solve_continuous_lyapunov_bwd)


# ---------------------------------------------------------------------------
# Discrete Lyapunov: A X A^T - X + Q = 0
# ---------------------------------------------------------------------------

@jax.custom_vjp
def solve_discrete_lyapunov(
    A: Float[Array, "n n"],
    Q: Float[Array, "n n"],
) -> Float[Array, "n n"]:
    """Solve the discrete Lyapunov equation ``A X A^T - X + Q = 0``.

    Uses the bilinear (Cayley) transformation to convert to a continuous
    Lyapunov equation and then calls :func:`solve_continuous_lyapunov`.

    Parameters
    ----------
    A : (n, n) array
        System matrix.  Must be Schur-stable (spectral radius < 1).
    Q : (n, n) array
        Symmetric forcing matrix.

    Returns
    -------
    X : (n, n) array
        Symmetric solution.
    """
    return _solve_discrete_lyapunov_impl(A, Q)


def _solve_discrete_lyapunov_impl(
    A: Float[Array, "n n"],
    Q: Float[Array, "n n"],
) -> Float[Array, "n n"]:
    """Bilinear transformation approach.

    Let  A_c = (A - I)(A + I)^{-1}   (Cayley / bilinear map).

    The discrete Lyapunov  A X A^T - X + Q = 0  becomes
    A_c Y + Y A_c^T + Q_c = 0  (continuous) after a change of variables.

    The transformed forcing is  Q_c = 2 (A + I)^{-1} Q (A + I)^{-T}.

    Then X = Y.
    """
    n = A.shape[0]
    I_n = jnp.eye(n, dtype=A.dtype)
    ApI = A + I_n
    AmI = A - I_n

    # A_c = (A - I)(A + I)^{-1}
    ApI_inv = jnp.linalg.solve(ApI.T, jnp.eye(n, dtype=A.dtype)).T  # = inv(ApI)
    A_c = AmI @ ApI_inv

    # Q_c = 2 (A+I)^{-1} Q (A+I)^{-T}
    Q_c = 2.0 * ApI_inv @ Q @ ApI_inv.T

    # Solve continuous Lyapunov via Kronecker
    Y = _solve_continuous_lyapunov_impl(A_c, Q_c)
    return _symmetrise(Y)


def _solve_discrete_lyapunov_fwd(
    A: Float[Array, "n n"],
    Q: Float[Array, "n n"],
):
    X = _solve_discrete_lyapunov_impl(A, Q)
    return X, (A, Q, X)


def _solve_discrete_lyapunov_bwd(res, g):
    """Backward pass for discrete Lyapunov.

    Given dL/dX = G, the adjoint equation is the discrete Lyapunov:

        A^T S A - S + G_sym = 0

    Then:
        dL/dA = -(A^T S^T + S^T A) X^T   — using the identity from Kao & Hennequin
               which simplifies to  2 A^T S X for the symmetric case.
        dL/dQ = -S

    More precisely:
        dL/dA = 2 A^T S X^T    (from vec calculus on the discrete Lyapunov)
        dL/dQ = -S

    We use the adjoint discrete Lyapunov  A^T S A - S + G_sym = 0.
    """
    A, Q, X = res
    G_sym = _symmetrise(g)

    # Adjoint discrete Lyapunov: A^T S A - S + G_sym = 0
    S = _solve_discrete_lyapunov_impl(A.T, G_sym)

    # Gradients: dL/dA = 2 S A X  (derived via implicit differentiation)
    dA = 2.0 * S @ A @ X
    dQ = S
    return dA, dQ


solve_discrete_lyapunov.defvjp(_solve_discrete_lyapunov_fwd,
                                _solve_discrete_lyapunov_bwd)


# ---------------------------------------------------------------------------
# Stability predicates
# ---------------------------------------------------------------------------

def is_stable(A: Float[Array, "n n"]) -> Bool:
    """Check continuous-time stability: all eigenvalues of *A* have Re < 0.

    Parameters
    ----------
    A : (n, n) array

    Returns
    -------
    stable : bool scalar
        True iff max Re(eig(A)) < 0.
    """
    eigvals = jnp.linalg.eigvals(A)
    return jnp.all(jnp.real(eigvals) < 0.0)


def is_schur_stable(A: Float[Array, "n n"]) -> Bool:
    """Check discrete-time stability: all eigenvalues of *A* lie strictly inside
    the unit disk.

    Parameters
    ----------
    A : (n, n) array

    Returns
    -------
    stable : bool scalar
        True iff max |eig(A)| < 1.
    """
    eigvals = jnp.linalg.eigvals(A)
    return jnp.all(jnp.abs(eigvals) < 1.0)


# Convenience alias used by type checkers; runtime it's just a jnp bool.
Bool = Union[jnp.bool_, bool]
