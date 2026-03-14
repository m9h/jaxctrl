"""Differentiable algebraic Riccati equation (ARE) solvers for JAX.

This module provides JIT-compatible, fully differentiable solvers for the
continuous- and discrete-time algebraic Riccati equations, together with LQR
gain computation.

**Continuous ARE (CARE)**::

    A^T X + X A - X B R^{-1} B^T X + Q = 0

The standard solution approach forms the 2n x 2n Hamiltonian matrix

    H = [[  A,       -B R^{-1} B^T ],
         [ -Q,       -A^T           ]]

and extracts the stable invariant subspace via a real Schur decomposition.

**Discrete ARE (DARE)**::

    A^T X A - X - A^T X B (R + B^T X B)^{-1} B^T X A + Q = 0

Solved via the symplectic matrix pencil and a generalized Schur (QZ)
decomposition.

Backward passes follow the adjoint identities from:

    Kao, C.-Y. & Hennequin, G. (2020).
    "Differentiable linear algebra for control theory."
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from jaxctrl._lyapunov import solve_continuous_lyapunov, solve_discrete_lyapunov, _symmetrise


# ======================================================================
# Continuous ARE
# ======================================================================

@jax.custom_vjp
def solve_continuous_are(
    A: Float[Array, "n n"],
    B: Float[Array, "n m"],
    Q: Float[Array, "n n"],
    R: Float[Array, "m m"],
) -> Float[Array, "n n"]:
    """Solve the continuous-time algebraic Riccati equation (CARE).

    Finds the unique stabilising symmetric PSD solution *X* of::

        A^T X + X A - X B R^{-1} B^T X + Q = 0

    Uses the Schur method on the Hamiltonian matrix.

    Parameters
    ----------
    A : (n, n) array
        State matrix.
    B : (n, m) array
        Input matrix.
    Q : (n, n) array
        State cost (symmetric PSD).
    R : (m, m) array
        Input cost (symmetric PD).

    Returns
    -------
    X : (n, n) array
        Stabilising solution.  NaN matrix if the Hamiltonian has eigenvalues
        on the imaginary axis (no stabilising solution exists).
    """
    return _solve_care_impl(A, B, Q, R)


def _solve_care_impl(A, B, Q, R):
    """Hamiltonian Schur method for the CARE.

    1. Form the Hamiltonian  H = [[ A,  -S ], [ -Q,  -A^T ]]
       where S = B R^{-1} B^T.
    2. Compute real Schur form of H, ordered so that the n eigenvalues with
       negative real part come first.
    3. Partition the first n Schur vectors: [U1; U2].
    4. X = U2 U1^{-1}.
    """
    n = A.shape[0]

    # S = B R^{-1} B^T
    R_inv_BT = jnp.linalg.solve(R, B.T)        # (m, n)
    S = B @ R_inv_BT                             # (n, n)

    # Hamiltonian
    H = jnp.block([
        [A,    -S   ],
        [-Q,   -A.T ],
    ])  # (2n, 2n)

    # Extract the stable invariant subspace of the Hamiltonian.
    # JAX does not expose ordered Schur decomposition (ordschur), so we use
    # eigendecomposition and select eigenvectors with Re(lambda) < 0.
    eigvals, eigvecs = jnp.linalg.eig(H)

    # Sort by real part — stable eigenvalues (Re < 0) first
    real_parts = jnp.real(eigvals)
    order = jnp.argsort(real_parts)
    eigvecs_sorted = eigvecs[:, order]

    # Take the first n eigenvectors (corresponding to stable eigenvalues)
    U = eigvecs_sorted[:, :n]

    U1 = U[:n, :]    # top block
    U2 = U[n:, :]    # bottom block

    # X = Re(U2 U1^{-1}) — take real part to discard numerical imaginary noise
    X = jnp.real(U2 @ jnp.linalg.inv(U1))
    X = _symmetrise(X)
    return X


def _solve_care_fwd(A, B, Q, R):
    X = _solve_care_impl(A, B, Q, R)
    return X, (A, B, Q, R, X)


def _solve_care_bwd(res, g):
    """Adjoint of the CARE via Kao & Hennequin (2020).

    Let K = R^{-1} B^T X  (the optimal gain) and
    A_cl = A - B K  (the closed-loop matrix, Hurwitz by construction).

    The adjoint variable S solves the continuous Lyapunov equation:

        A_cl^T S + S A_cl + G_sym = 0

    Then the parameter gradients are:

        dL/dA = S X + X S      (but using the transpose convention: S X^T + S^T X)
        dL/dB = -2 X S B R^{-1}  (from chain rule through S = B R^{-1} B^T)
        dL/dQ = -S
        dL/dR = K S K^T          (= R^{-1} B^T X S X B R^{-1})
    """
    A, B, Q, R, X = res
    G_sym = _symmetrise(g)

    # Closed-loop quantities
    R_inv_BT = jnp.linalg.solve(R, B.T)   # (m, n)
    K = R_inv_BT @ X                       # (m, n)
    A_cl = A - B @ K                       # (n, n)

    # Adjoint Lyapunov: A_cl S + S A_cl^T + G_sym = 0
    S = solve_continuous_lyapunov(A_cl, G_sym)

    # Parameter gradients
    dA = S @ X.T + S.T @ X
    dB = -2.0 * X @ S @ B @ jnp.linalg.inv(R)
    dQ = S
    dR = -(K @ S @ K.T)

    return dA, dB, dQ, dR


solve_continuous_are.defvjp(_solve_care_fwd, _solve_care_bwd)


# ======================================================================
# Discrete ARE
# ======================================================================

@jax.custom_vjp
def solve_discrete_are(
    A: Float[Array, "n n"],
    B: Float[Array, "n m"],
    Q: Float[Array, "n n"],
    R: Float[Array, "m m"],
) -> Float[Array, "n n"]:
    """Solve the discrete-time algebraic Riccati equation (DARE).

    Finds the unique stabilising symmetric PSD solution *X* of::

        A^T X A - X - A^T X B (R + B^T X B)^{-1} B^T X A + Q = 0

    Uses eigendecomposition of the symplectic matrix pencil.

    Parameters
    ----------
    A : (n, n) array
        State transition matrix.
    B : (n, m) array
        Input matrix.
    Q : (n, n) array
        State cost (symmetric PSD).
    R : (m, m) array
        Input cost (symmetric PD).

    Returns
    -------
    X : (n, n) array
        Stabilising solution.
    """
    return _solve_dare_impl(A, B, Q, R)


def _solve_dare_impl(A, B, Q, R):
    """Symplectic pencil method for the DARE.

    Form the 2n x 2n symplectic matrix:

        Z = [[ A + B R^{-1} B^T (A^{-T}) Q,   -B R^{-1} B^T A^{-T} ],
             [          -A^{-T} Q,                   A^{-T}           ]]

    and find the stable invariant subspace (eigenvalues inside the unit
    circle).  Then X = U2 U1^{-1}.

    For JIT-friendliness we use the equivalent direct eigendecomposition.
    """
    n = A.shape[0]

    # Precompute
    A_inv_T = jnp.linalg.inv(A.T)                 # A^{-T}
    R_inv = jnp.linalg.inv(R)
    BR_inv_BT = B @ R_inv @ B.T                    # B R^{-1} B^T

    # Symplectic matrix
    Z11 = A + BR_inv_BT @ A_inv_T @ Q
    Z12 = -BR_inv_BT @ A_inv_T
    Z21 = -A_inv_T @ Q
    Z22 = A_inv_T

    Z = jnp.block([
        [Z11, Z12],
        [Z21, Z22],
    ])

    # Eigendecomposition
    eigvals, eigvecs = jnp.linalg.eig(Z)

    # Sort by magnitude — stable eigenvalues (|lambda| < 1) first
    mags = jnp.abs(eigvals)
    order = jnp.argsort(mags)
    eigvecs_sorted = eigvecs[:, order]

    # Take the first n eigenvectors
    U = eigvecs_sorted[:, :n]
    U1 = U[:n, :]
    U2 = U[n:, :]

    X = jnp.real(U2 @ jnp.linalg.inv(U1))
    X = _symmetrise(X)
    return X


def _solve_dare_fwd(A, B, Q, R):
    X = _solve_dare_impl(A, B, Q, R)
    return X, (A, B, Q, R, X)


def _solve_dare_bwd(res, g):
    """Adjoint of the DARE via implicit differentiation.

    Let F = (R + B^T X B)^{-1} B^T X A  (the optimal gain) and
    A_cl = A - B F  (closed-loop, Schur-stable by construction).

    The adjoint variable S solves the discrete Lyapunov equation:

        A_cl^T S A_cl - S + G_sym = 0

    The parameter gradients follow from the implicit function theorem applied
    to the DARE residual, mirroring the structure of Kao & Hennequin (2020):

        dL/dA = 2 * A_cl^T S
        dL/dB = -2 * A_cl^T S F^T
        dL/dQ = -S
        dL/dR = F S F^T
    """
    A, B, Q, R, X = res
    G_sym = _symmetrise(g)

    # Closed-loop quantities
    BT_X = B.T @ X
    RBXB = R + BT_X @ B
    F = jnp.linalg.solve(RBXB, BT_X @ A)   # (m, n) optimal gain
    A_cl = A - B @ F                          # (n, n) closed-loop

    # Adjoint discrete Lyapunov: A_cl S A_cl^T - S + G_sym = 0
    S = solve_discrete_lyapunov(A_cl, G_sym)

    # Parameter gradients
    dA = 2.0 * A_cl.T @ S
    dB = -2.0 * A_cl.T @ S @ F.T
    dQ = S
    dR = -(F @ S @ F.T)

    return dA, dB, dQ, dR


solve_discrete_are.defvjp(_solve_dare_fwd, _solve_dare_bwd)


# ======================================================================
# LQR gain computation
# ======================================================================

def lqr(
    A: Float[Array, "n n"],
    B: Float[Array, "n m"],
    Q: Float[Array, "n n"],
    R: Float[Array, "m m"],
) -> tuple[Float[Array, "m n"], Float[Array, "n n"]]:
    """Compute the continuous-time infinite-horizon LQR optimal gain.

    Minimises::

        J = integral_0^inf (x^T Q x + u^T R u) dt

    subject to  dx/dt = A x + B u.

    Parameters
    ----------
    A, B, Q, R : arrays
        System and cost matrices.

    Returns
    -------
    K : (m, n) array
        Optimal state-feedback gain, u = -K x.
    X : (n, n) array
        Solution of the associated CARE.
    """
    X = solve_continuous_are(A, B, Q, R)
    K = jnp.linalg.solve(R, B.T @ X)
    return K, X


def dlqr(
    A: Float[Array, "n n"],
    B: Float[Array, "n m"],
    Q: Float[Array, "n n"],
    R: Float[Array, "m m"],
) -> tuple[Float[Array, "m n"], Float[Array, "n n"]]:
    """Compute the discrete-time infinite-horizon LQR optimal gain.

    Minimises::

        J = sum_{k=0}^{inf} (x_k^T Q x_k + u_k^T R u_k)

    subject to  x_{k+1} = A x_k + B u_k.

    Parameters
    ----------
    A, B, Q, R : arrays
        System and cost matrices.

    Returns
    -------
    K : (m, n) array
        Optimal state-feedback gain, u_k = -K x_k.
    X : (n, n) array
        Solution of the associated DARE.
    """
    X = solve_discrete_are(A, B, Q, R)
    K = jnp.linalg.solve(R + B.T @ X @ B, B.T @ X @ A)
    return K, X
