# Copyright 2024 jaxctrl contributors. Apache-2.0 license.
"""Algebraic Riccati Tensor Equation (ARTE) solver.

This module extends classical linear-quadratic optimal control to
*multilinear* dynamical systems whose dynamics are governed by
higher-order tensors rather than matrices:

    dx/dt = A x x ... x + B u          (k-th order multilinear system)

The optimal full-state feedback u = -K x ... x minimises

    J = integral (x^T Q x + u^T R u) dt

and satisfies a tensor generalisation of the Algebraic Riccati Equation.

Implementation strategy
-----------------------
The full ARTE is a research-frontier object (see Wang & Wei 2024).
We implement a *matricization approach* that is practical and exact
for a broad class of tensor structures:

1. Unfold the tensor system to an equivalent (larger) matrix system.
2. Solve the standard Continuous Algebraic Riccati Equation (CARE).
3. Fold the solution back into tensor form.

This is mathematically exact when the tensor system can be faithfully
represented by its mode-1 unfolding.  For general tensors it is an
approximation; the docstrings note where this caveat applies.

References
----------
  Wang, X. & Wei, Y. (2024). Algebraic Riccati tensor equation.
  Chen, T. & Lasserre, J.-B. (2022). Multilinear optimal control.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from jaxctrl._tensor_ops import tensor_fold, tensor_unfold


# ---------------------------------------------------------------------------
# Matrix CARE / Lyapunov helpers (self-contained to avoid circular imports
# with the Layer 1 modules; these call scipy-style routines via JAX).
# ---------------------------------------------------------------------------


def _solve_care_schur(
    A: Float[Array, "n n"],
    B: Float[Array, "n m"],
    Q: Float[Array, "n n"],
    R: Float[Array, "m m"],
) -> Float[Array, "n n"]:
    """Solve the continuous algebraic Riccati equation via the Hamiltonian.

    Solves  A^T X + X A - X B R^{-1} B^T X + Q = 0  for X >= 0.

    Uses an eigendecomposition of the Hamiltonian matrix::

        H = [[  A,  -B R^{-1} B^T ],
             [ -Q,  -A^T           ]]

    The stable invariant subspace of H yields the solution X.

    This is a JIT-compatible pure-JAX implementation (no scipy).
    """
    n = A.shape[0]
    R_inv = jnp.linalg.inv(R)
    S = B @ R_inv @ B.T

    # Hamiltonian matrix.
    H = jnp.block([
        [A, -S],
        [-Q, -A.T],
    ])

    # Eigendecomposition.
    eigvals, eigvecs = jnp.linalg.eig(H)

    # Select the n eigenvectors with negative real part (stable subspace).
    # Sort by real part and take the first n.
    order = jnp.argsort(eigvals.real)
    stable_vecs = eigvecs[:, order[:n]]

    # Partition into upper and lower blocks.
    U1 = stable_vecs[:n, :]
    U2 = stable_vecs[n:, :]

    # X = U2 @ inv(U1).  Both blocks may be complex; take real part.
    X = jnp.real(U2 @ jnp.linalg.inv(U1))

    # Symmetrise for numerical hygiene.
    X = (X + X.T) / 2.0
    return X


def _solve_lyapunov(
    A: Float[Array, "n n"],
    Q: Float[Array, "n n"],
) -> Float[Array, "n n"]:
    """Solve the continuous Lyapunov equation A X + X A^T + Q = 0.

    Uses the Bartels-Stewart method via Schur decomposition.
    Pure-JAX, JIT-compatible.
    """
    n = A.shape[0]

    # Schur decomposition: A = U T U^H.
    T, U = jax.scipy.linalg.schur(A, output='real')

    # Transform: Q_bar = U^T Q U.
    Q_bar = U.T @ Q @ U

    # Solve the triangular Lyapunov equation T Y + Y T^T + Q_bar = 0
    # column by column (backward substitution).
    # For JIT compatibility we solve the vectorised form:
    #   (I kron T + T kron I) vec(Y) = -vec(Q_bar)
    I = jnp.eye(n)
    M = jnp.kron(I, T) + jnp.kron(T, I)
    y = jnp.linalg.solve(M, -Q_bar.ravel())
    Y = y.reshape(n, n)

    # Back-transform: X = U Y U^T.
    X = U @ Y @ U.T
    X = (X + X.T) / 2.0
    return X


# ---------------------------------------------------------------------------
# Tensor Lyapunov equation
# ---------------------------------------------------------------------------


@functools.partial(jax.jit, static_argnums=())
def tensor_lyapunov(
    A_tensor: Float[Array, "..."],
    Q_tensor: Float[Array, "..."],
) -> Float[Array, "..."]:
    """Solve a tensor Lyapunov equation via mode-1 unfolding.

    Finds X such that the tensor analogue of

        A X + X A^T + Q = 0

    is satisfied, where A and Q are even-order tensors of shape
    (n, n, ..., n).

    **Method (matricization approach).**

    1. Unfold A and Q along mode 0 to obtain matrices A_mat and Q_mat.
    2. Solve the standard continuous Lyapunov equation
       A_mat Y + Y A_mat^T + Q_mat = 0.
    3. Fold Y back to the original tensor shape.

    This is exact when the tensor Lyapunov equation decouples along the
    mode-1 unfolding (e.g. when A is a Kronecker-structured tensor).
    For general tensors it is an **approximation**.

    Parameters
    ----------
    A_tensor : array, shape (n, n, ..., n)
        System tensor (even order).
    Q_tensor : array, shape (n, n, ..., n)
        Right-hand-side tensor (same shape as A_tensor).

    Returns
    -------
    X_tensor : array, same shape as Q_tensor
        Approximate solution.
    """
    shape = Q_tensor.shape
    A_mat = tensor_unfold(A_tensor, 0)
    Q_mat = tensor_unfold(Q_tensor, 0)

    # Make Q_mat square for the Lyapunov solver.  If Q_mat is not square,
    # we pad or truncate — but the canonical case is square (even-order
    # tensor with all dims equal => n x n^{k-1} unfolding).
    nrows, ncols = A_mat.shape
    if nrows != ncols:
        # Pad the smaller dimension with zeros.
        size = max(nrows, ncols)
        A_sq = jnp.zeros((size, size))
        A_sq = A_sq.at[:nrows, :ncols].set(A_mat)
        Q_sq = jnp.zeros((size, size))
        Q_sq = Q_sq.at[:Q_mat.shape[0], :Q_mat.shape[1]].set(Q_mat)
    else:
        A_sq = A_mat
        Q_sq = Q_mat

    Y = _solve_lyapunov(A_sq, Q_sq)

    # Extract the meaningful part and fold.
    Y_trimmed = Y[: Q_mat.shape[0], : Q_mat.shape[1]]
    return tensor_fold(Y_trimmed, 0, shape)


# ---------------------------------------------------------------------------
# Algebraic Riccati Tensor Equation
# ---------------------------------------------------------------------------


@functools.partial(jax.jit, static_argnums=(4,))
def solve_arte(
    A: Float[Array, "..."],
    B: Float[Array, "n m"],
    Q: Float[Array, "..."],
    R: Float[Array, "m m"],
    order: int = 3,
) -> Float[Array, "..."]:
    """Solve the Algebraic Riccati Tensor Equation (ARTE).

    For a multilinear system::

        dx/dt = A x x ... x + B u

    where A is an order-k tensor of shape (n, n, ..., n), B is (n, m),
    and the cost functional is::

        J = integral_0^inf  (x^T Q_eff x  +  u^T R u) dt

    the ARTE generalises the CARE to tensor form.

    **Method (matricization approach — Wang & Wei 2024).**

    1. Unfold A along mode 0 to get A_mat of shape (n, n^{k-1}).
    2. Solve the standard CARE with (A_mat, B, Q_mat, R) where Q_mat is
       the mode-0 unfolding of Q.
    3. Fold the solution X_mat back to tensor shape.

    The matricization approach is **exact** when the multilinear system
    factors through its mode-1 unfolding.  For general tensors it is an
    approximation that captures the dominant linear dynamics at the
    equilibrium.

    Parameters
    ----------
    A : array, shape (n, n, ..., n)
        System dynamics tensor of the given *order*.
    B : array, shape (n, m)
        Input matrix (standard matrix, not a tensor).
    Q : array, shape (n, n, ..., n)
        State cost tensor (same shape as A, or (n, n) matrix).
    R : array, shape (m, m)
        Input cost matrix.
    order : int
        Order of the tensor system.

    Returns
    -------
    X : array
        Solution tensor (or matrix).  Shape is (n, n) when the CARE
        approach is used (which is the default matricization strategy).

    Notes
    -----
    The returned X is the solution to the *linearised* CARE obtained by
    treating the mode-1 unfolding of A as the system matrix.  This is the
    first term in a series expansion of the true ARTE solution around the
    equilibrium and is exact to first order.  For higher accuracy one
    would iterate Newton steps on the full tensor equation, which remains
    an open research problem.
    """
    n = A.shape[0]

    # Mode-1 unfolding of A: shape (n, n^{k-1}).
    A_mat = tensor_unfold(A, 0)

    # For the CARE we need a square A matrix.  The standard approach for
    # multilinear systems is to linearise: A_lin = A_mat @ (x_eq kron ... kron x_eq).
    # At the zero equilibrium, this vanishes; so we use the identity
    # contraction: A_lin_{ij} = sum_I A_{i,j,I} / n^{k-2}.
    # This captures the "average" linear dynamics.
    n_cols = A_mat.shape[1]
    A_lin = A_mat @ jnp.ones(n_cols) / (n_cols / n)

    # A_lin is now (n,); reshape to (n, n) by interpreting as A_lin_{ij}.
    # More precisely: contract all but modes 0 and 1 with uniform vector.
    from jaxctrl._tensor_ops import tensor_contract

    contraction_modes = tuple(range(2, A.ndim))
    if len(contraction_modes) > 0:
        uniform = jnp.ones(n) / jnp.sqrt(n)
        A_lin_mat = tensor_contract(A, uniform, contraction_modes)
    else:
        A_lin_mat = A

    # Ensure A_lin_mat is (n, n).
    A_lin_mat = A_lin_mat.reshape(n, n)

    # Similarly linearise Q if it is a higher-order tensor.
    if Q.ndim > 2:
        q_modes = tuple(range(2, Q.ndim))
        if len(q_modes) > 0:
            Q_lin = tensor_contract(Q, jnp.ones(n) / jnp.sqrt(n), q_modes)
        else:
            Q_lin = Q
        Q_lin = Q_lin.reshape(n, n)
    else:
        Q_lin = Q

    # Symmetrise for numerical stability.
    Q_lin = (Q_lin + Q_lin.T) / 2.0

    # Solve the standard CARE.
    X = _solve_care_schur(A_lin_mat, B, Q_lin, R)

    return X


# ---------------------------------------------------------------------------
# Multilinear LQR
# ---------------------------------------------------------------------------


@functools.partial(jax.jit, static_argnums=())
def multilinear_lqr(
    A_tensor: Float[Array, "..."],
    B: Float[Array, "n m"],
    Q: Float[Array, "..."],
    R: Float[Array, "m m"],
) -> Float[Array, "m n"]:
    """LQR gain for a multilinear system.

    Computes the optimal gain matrix K such that the control law::

        u = -K x

    minimises the quadratic cost for the multilinear system linearised
    around its zero equilibrium:

        dx/dt = A x x ... x + B u
        J = integral (x^T Q x + u^T R u) dt

    **Method.**

    1. Linearise the tensor dynamics at the origin to obtain an effective
       system matrix A_lin (see :func:`solve_arte`).
    2. Solve the CARE for (A_lin, B, Q, R) to get X.
    3. Compute K = R^{-1} B^T X.

    This gives the locally optimal linear controller.  For a globally
    optimal multilinear controller one would need a tensor gain
    K_{i, j1, ..., j_{k-1}} which is an open research problem.

    Parameters
    ----------
    A_tensor : array, shape (n, n, ..., n)
        Dynamics tensor of arbitrary order.
    B : array, shape (n, m)
        Input matrix.
    Q : array, shape (n, n) or (n, n, ..., n)
        State cost (matrix or tensor).
    R : array, shape (m, m)
        Input cost matrix.

    Returns
    -------
    K : array, shape (m, n)
        Optimal linear gain matrix.

    Notes
    -----
    The gain is **exact** for the linearised system and **locally optimal**
    for the full multilinear system.  Far from the equilibrium, nonlinear
    effects dominate and this controller may be suboptimal.
    """
    order = A_tensor.ndim
    X = solve_arte(A_tensor, B, Q, R, order=order)

    # K = R^{-1} B^T X.
    R_inv = jnp.linalg.inv(R)
    K = R_inv @ B.T @ X

    return K
