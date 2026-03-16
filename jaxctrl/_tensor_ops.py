# Copyright 2024 jaxctrl contributors. Apache-2.0 license.
"""Tensor algebra utilities for control theory on hypergraphs.

This module provides the core tensor operations needed when adjacency
structure is described by a higher-order tensor rather than a matrix.
All functions are pure-functional, JIT-compatible, and differentiable
via JAX autodiff.

Mathematical background
-----------------------
In a k-uniform hypergraph on n nodes the adjacency structure is captured
by an order-k tensor A of shape (n, ..., n).  Standard linear-algebraic
control operations (matrix products, eigendecompositions, Lyapunov
equations) must be lifted to *tensor* analogues.  This module supplies
the building blocks.

References
----------
  Qi, L. (2005). Eigenvalues of a real supersymmetric tensor.
  Kolda, T. & Bader, B. (2009). Tensor decompositions and applications.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


# ---------------------------------------------------------------------------
# Einstein product
# ---------------------------------------------------------------------------


@functools.partial(jax.jit, static_argnums=(2,))
def einstein_product(
    A: Float[Array, "..."],
    B: Float[Array, "..."],
    modes: int,
) -> Float[Array, "..."]:
    """Einstein product of two tensors.

    The Einstein product contracts the last *modes* axes of ``A`` with the
    first *modes* axes of ``B``.  For matrices (order-2 tensors) with
    ``modes=1`` this reduces to standard matrix multiplication.

    Parameters
    ----------
    A : array
        Left tensor of arbitrary order.
    B : array
        Right tensor of arbitrary order.
    modes : int
        Number of axes to contract.  The last ``modes`` axes of *A* are
        summed against the first ``modes`` axes of *B*.  These axes must
        have matching lengths.

    Returns
    -------
    array
        Result tensor of order ``ndim(A) + ndim(B) - 2 * modes``.
    """
    ndim_a = A.ndim
    ndim_b = B.ndim

    # Build einsum index strings.
    # A indices: 0 .. ndim_a-1
    # B indices: (ndim_a - modes) .. (ndim_a - modes + ndim_b - 1)
    #   where the first `modes` of B reuse the last `modes` of A.
    a_idx = list(range(ndim_a))
    b_idx = list(range(ndim_a - modes, ndim_a - modes + ndim_b))

    # Output: free indices of A then free indices of B
    out_idx = a_idx[: ndim_a - modes] + b_idx[modes:]

    return jnp.einsum(A, a_idx, B, b_idx, out_idx)


# ---------------------------------------------------------------------------
# Mode-n unfolding / folding
# ---------------------------------------------------------------------------


@functools.partial(jax.jit, static_argnums=(1,))
def tensor_unfold(
    T: Float[Array, "..."],
    mode: int,
) -> Float[Array, "n rest"]:
    """Mode-*n* unfolding (matricization) of a tensor.

    Rearranges the elements of an order-k tensor into a matrix whose rows
    correspond to the fibers along *mode* and whose columns cycle through
    the remaining modes in the natural order.

    Parameters
    ----------
    T : array, shape (n_0, n_1, ..., n_{k-1})
        Input tensor.
    mode : int
        The mode along which to unfold (0-indexed).

    Returns
    -------
    array, shape (n_mode, prod(n_i for i != mode))
        The unfolded matrix.
    """
    # Move the target mode to the front, then flatten all other dims.
    return jnp.reshape(jnp.moveaxis(T, mode, 0), (T.shape[mode], -1))


@functools.partial(jax.jit, static_argnums=(1, 2))
def tensor_fold(
    M: Float[Array, "n rest"],
    mode: int,
    shape: tuple[int, ...],
) -> Float[Array, "..."]:
    """Inverse of :func:`tensor_unfold`.

    Parameters
    ----------
    M : array, shape (n_mode, prod(n_i for i != mode))
        Unfolded matrix.
    mode : int
        The mode that corresponds to the rows of *M*.
    shape : tuple of int
        The desired tensor shape.

    Returns
    -------
    array of the given *shape*.
    """
    # Reconstruct the transposed shape: mode-dim first, then the rest.
    remaining_shape = list(shape[:mode]) + list(shape[mode + 1 :])
    T = jnp.reshape(M, [shape[mode]] + remaining_shape)
    return jnp.moveaxis(T, 0, mode)


# ---------------------------------------------------------------------------
# Tensor contraction with a vector
# ---------------------------------------------------------------------------


@functools.partial(jax.jit, static_argnums=(2,))
def tensor_contract(
    T: Float[Array, "..."],
    v: Float[Array, " n"],
    modes: tuple[int, ...],
) -> Float[Array, "..."]:
    """Contract a tensor with a vector along specified modes.

    For an order-k adjacency tensor A and a state vector x, the
    multilinear dynamics vector field is obtained by contracting A with x
    along all modes except the first:

        f(x) = A x_{i2} x_{i3} ... x_{ik}

    which is ``tensor_contract(A, x, modes=(1, 2, ..., k-1))``.

    Parameters
    ----------
    T : array, shape (n, n, ..., n)
        Input tensor of arbitrary order.
    v : array, shape (n,)
        Vector to contract with.
    modes : tuple of int
        Axes of *T* along which to contract with *v*.  Contractions are
        applied sequentially from highest to lowest mode index so that
        earlier indices remain valid.

    Returns
    -------
    array
        Resulting tensor with ``len(modes)`` fewer dimensions.
    """
    result = T
    # Sort modes descending so that removing higher axes first keeps
    # lower axis indices valid.
    for m in sorted(modes, reverse=True):
        result = jnp.tensordot(result, v, axes=([m], [0]))
    return result


# ---------------------------------------------------------------------------
# Symmetrization
# ---------------------------------------------------------------------------


@jax.jit
def symmetrize_tensor(
    T: Float[Array, "..."],
) -> Float[Array, "..."]:
    """Symmetrize a tensor by averaging over all permutations of its indices.

    A tensor is *supersymmetric* if it is invariant under any permutation
    of its indices.  For a hypergraph adjacency tensor, symmetry encodes
    the fact that a hyperedge {i, j, k} is the same regardless of the
    ordering of its vertices.

    Parameters
    ----------
    T : array, shape (n, n, ..., n)
        Input tensor.  All dimensions must be equal.

    Returns
    -------
    array
        The symmetrized tensor, same shape as *T*.

    Notes
    -----
    For order k the number of permutations is k!.  This is exact but
    expensive for large k.  For k <= 6 it is fine.
    """
    order = T.ndim
    axes = list(range(order))
    perms = _permutations(axes)
    acc = jnp.zeros_like(T)
    for p in perms:
        acc = acc + jnp.transpose(T, p)
    return acc / len(perms)


def _permutations(seq: list[int]) -> list[tuple[int, ...]]:
    """Return all permutations of *seq* (small helper, not JIT-traced)."""
    if len(seq) <= 1:
        return [tuple(seq)]
    result: list[tuple[int, ...]] = []
    for i, elem in enumerate(seq):
        rest = seq[:i] + seq[i + 1 :]
        for perm in _permutations(rest):
            result.append((elem, *perm))
    return result


# ---------------------------------------------------------------------------
# Khatri-Rao product
# ---------------------------------------------------------------------------


@jax.jit
def khatri_rao(
    A: Float[Array, "i c"],
    B: Float[Array, "j c"],
) -> Float[Array, "ij c"]:
    """Khatri-Rao product (column-wise Kronecker product).

    Given A of shape (I, C) and B of shape (J, C), returns a matrix of
    shape (I*J, C) where each column is the Kronecker product of the
    corresponding columns of A and B.

    This arises in CP / PARAFAC tensor decomposition and is used when
    reconstructing tensors from factor matrices.

    Parameters
    ----------
    A : array, shape (I, C)
    B : array, shape (J, C)

    Returns
    -------
    array, shape (I*J, C)
    """
    # A[:, None, :] has shape (I, 1, C)
    # B[None, :, :] has shape (1, J, C)
    # Element-wise product gives (I, J, C), reshape to (I*J, C).
    I, C = A.shape
    J, _ = B.shape
    return (A[:, None, :] * B[None, :, :]).reshape(I * J, C)


# ---------------------------------------------------------------------------
# n-mode product
# ---------------------------------------------------------------------------


@functools.partial(jax.jit, static_argnums=(2,))
def mode_dot(
    tensor: Float[Array, "..."],
    matrix_or_vector: Float[Array, "..."],
    mode: int,
) -> Float[Array, "..."]:
    """n-mode product of a tensor with a matrix or vector.

    The n-mode product of tensor T with matrix U along mode *n*
    replaces the mode-*n* fibers by U times those fibers:

        (T x_n U)_{i_0 ... i_{n-1} j i_{n+1} ...}
            = sum_k T_{i_0 ... k ... } U_{j,k}

    For a vector v (1-D) the result has one fewer dimension (contraction).

    Parameters
    ----------
    tensor : array
        Input tensor.
    matrix_or_vector : array
        If 2-D (J, I_n): replaces dimension n of size I_n with size J.
        If 1-D (I_n,): contracts along mode n.
    mode : int
        The mode along which to apply the product.

    Returns
    -------
    array
    """
    if matrix_or_vector.ndim == 2:
        res = jnp.tensordot(tensor, matrix_or_vector, axes=(mode, 1))
        return jnp.moveaxis(res, -1, mode)
    return jnp.tensordot(tensor, matrix_or_vector, axes=(mode, 0))


# ---------------------------------------------------------------------------
# Tucker decomposition (HOSVD)
# ---------------------------------------------------------------------------


def hosvd(
    tensor: Float[Array, "..."],
    ranks: list[int] | None = None,
) -> tuple[Float[Array, "..."], list[Float[Array, "..."]]]:
    """Higher-Order Singular Value Decomposition (HOSVD).

    Computes the Tucker decomposition via truncated SVD on each
    mode-unfolding.

    Parameters
    ----------
    tensor : array
        Input tensor of arbitrary order.
    ranks : list of int, optional
        Target rank for each mode.  If None, uses full rank.

    Returns
    -------
    core : array
        Core tensor (compressed).
    factors : list of array
        Factor matrices [U_0, U_1, ...] with shapes (n_i, r_i).
    """
    n_modes = tensor.ndim
    if ranks is None:
        ranks = list(tensor.shape)

    factors = []
    for m in range(n_modes):
        unfolded = tensor_unfold(tensor, m)
        U, _, _ = jnp.linalg.svd(unfolded, full_matrices=False)
        factors.append(U[:, : ranks[m]])

    # Core = tensor projected onto factor bases
    core = tensor
    for i, factor in enumerate(factors):
        core = mode_dot(core, factor.T, i)

    return core, factors


def tucker_to_tensor(
    core: Float[Array, "..."],
    factors: list[Float[Array, "..."]],
) -> Float[Array, "..."]:
    """Reconstruct tensor from Tucker decomposition (core, factors).

    Parameters
    ----------
    core : array
        Core tensor.
    factors : list of array
        Factor matrices [U_0, U_1, ...].

    Returns
    -------
    array
        Full tensor = core x_0 U_0 x_1 U_1 ...
    """
    tensor = core
    for i, factor in enumerate(factors):
        tensor = mode_dot(tensor, factor, i)
    return tensor


# ---------------------------------------------------------------------------
# Tensor trace
# ---------------------------------------------------------------------------


@functools.partial(jax.jit, static_argnums=())
def tensor_trace(
    T: Float[Array, "..."],
) -> Float[Array, ""]:
    """Generalized trace of a tensor.

    For an order-k tensor of shape (n, n, ..., n), the trace sums all
    elements whose indices are all equal:

        trace(T) = sum_{i=0}^{n-1} T[i, i, ..., i]

    For a matrix this coincides with the usual trace.

    Parameters
    ----------
    T : array, shape (n, n, ..., n)
        Input tensor.  All dimensions must be equal.

    Returns
    -------
    scalar
        The trace value.
    """
    n = T.shape[0]
    # Build the diagonal index array.
    idx = jnp.arange(n)
    # T[idx, idx, ..., idx]  (k copies)
    indices = tuple(idx for _ in range(T.ndim))
    return jnp.sum(T[indices])
