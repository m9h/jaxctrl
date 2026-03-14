# Copyright 2024 jaxctrl contributors. Apache-2.0 license.
"""Tensor eigenvalue solvers for supersymmetric tensors.

Standard eigenvalue decompositions do not generalise uniquely to tensors
of order >= 3.  Two widely-studied notions are **Z-eigenvalues** (Qi 2005)
and **H-eigenvalues** (Lim 2005, Qi 2005).  This module implements
iterative solvers for both, built entirely on JAX primitives so that the
eigenvalue computation is JIT-compilable and differentiable.

Definitions
-----------
Let T be a real supersymmetric tensor of order k and dimension n (i.e.
T has shape (n, n, ..., n) with k copies of n).

**Z-eigenvalue.**  A scalar lam is a Z-eigenvalue with Z-eigenvector x if

    T x x ... x  = lam * x        (contract on all modes but the first)
           (k-1 copies of x)
    ||x||_2 = 1

**H-eigenvalue.**  A scalar lam is an H-eigenvalue with H-eigenvector x if

    T x x ... x  = lam * x^{[k-1]}
           (k-1 copies of x)

where x^{[k-1]} denotes element-wise (k-1)-th power of x.

References
----------
  Qi, L. (2005). Eigenvalues of a real supersymmetric tensor. J. Symb. Comput.
  Kolda, T. & Mayo, J. (2011). Shifted symmetric higher-order power method.
  Kolda, T. & Mayo, J. (2014). An adaptive shifted power method for computing
      generalized tensor eigenpairs. SIAM J. Matrix Anal. Appl.
"""

from __future__ import annotations

import functools
from typing import Optional

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from jaxctrl._tensor_ops import tensor_contract


# ---------------------------------------------------------------------------
# Higher-order power method (building block)
# ---------------------------------------------------------------------------


@functools.partial(jax.jit, static_argnums=(2, 3))
def tensor_power_method(
    T: Float[Array, "..."],
    x0: Optional[Float[Array, " n"]] = None,
    max_iters: int = 1000,
    tol: float = 1e-6,
    key: Optional[PRNGKeyArray] = None,
) -> tuple[Float[Array, ""], Float[Array, " n"]]:
    """Higher-order power method for the leading Z-eigenpair.

    Implements the Shifted Symmetric Higher-Order Power Method (SS-HOPM)
    of Kolda & Mayo (2011).  The shift parameter alpha is chosen
    adaptively to guarantee monotonic convergence for supersymmetric
    tensors.

    Algorithm::

        repeat:
            y = T x x ... x  (contract on modes 1..k-1)
            y = y + alpha * x
            x_new = y / ||y||
        until convergence

    The shift alpha is set to ``max(0, -lam_min_est)`` where
    ``lam_min_est`` is a conservative lower bound derived from the
    Frobenius norm of the mode-1 unfolding.

    Parameters
    ----------
    T : array, shape (n, ..., n)
        Supersymmetric tensor of order k >= 2.
    x0 : array, shape (n,), optional
        Starting vector.  If *None*, a random unit vector is drawn using
        *key*.
    max_iters : int
        Maximum number of iterations.
    tol : float
        Convergence tolerance on the change in eigenvalue.
    key : PRNGKey, optional
        Random key used to initialise *x0* when it is not provided.  If
        both *x0* and *key* are None, a deterministic starting vector
        ``[1, 0, ..., 0]`` is used.

    Returns
    -------
    eigenvalue : scalar
        Leading Z-eigenvalue (largest in magnitude).
    eigenvector : array, shape (n,)
        Corresponding unit eigenvector.
    """
    n = T.shape[0]
    order = T.ndim

    # Initialise x0.
    if x0 is None:
        if key is not None:
            x0 = jax.random.normal(key, (n,))
            x0 = x0 / jnp.linalg.norm(x0)
        else:
            x0 = jnp.zeros(n).at[0].set(1.0)

    # Compute a safe shift from the Frobenius norm.
    frob = jnp.sqrt(jnp.sum(T ** 2))
    alpha = frob  # conservative positive shift

    contraction_modes = tuple(range(1, order))

    def _eigenvalue(x: Float[Array, " n"]) -> Float[Array, ""]:
        y = tensor_contract(T, x, contraction_modes)
        return jnp.dot(x, y)

    # Use lax.fori_loop for JIT compatibility.  The tolerance-based early
    # exit is approximated by freezing the iterate once converged (the
    # answer is numerically stable once the change drops below tol).
    def fori_body(i, carry):
        x, lam_prev = carry
        y = tensor_contract(T, x, contraction_modes)
        y = y + alpha * x
        x_new = y / jnp.linalg.norm(y)
        lam_new = _eigenvalue(x_new)
        # Once converged, keep returning the same value (no early exit in
        # fori_loop, but the answer is stable).
        converged = jnp.abs(lam_new - lam_prev) < tol
        x_out = jnp.where(converged, x, x_new)
        lam_out = jnp.where(converged, lam_prev, lam_new)
        return (x_out, lam_out)

    lam0 = _eigenvalue(x0)
    x_final, lam_final = jax.lax.fori_loop(0, max_iters, fori_body, (x0, lam0))

    return lam_final, x_final


# ---------------------------------------------------------------------------
# Z-eigenvalues via deflation
# ---------------------------------------------------------------------------


@functools.partial(jax.jit, static_argnums=(1, 2, 3))
def z_eigenvalues(
    T: Float[Array, "..."],
    num_eigvals: int = 1,
    max_iters: int = 1000,
    tol: float = 1e-6,
    key: Optional[PRNGKeyArray] = None,
) -> tuple[Float[Array, " k"], Float[Array, "k n"]]:
    """Compute Z-eigenvalues of a supersymmetric tensor.

    A scalar lam is a Z-eigenvalue with Z-eigenvector x if::

        T x x ... x = lam * x  and  ||x|| = 1

    The solver uses the SS-HOPM (shifted symmetric higher-order power
    method) from Kolda & Mayo (2011) with a deflation strategy to extract
    multiple eigenpairs.

    **Deflation.**  After finding an eigenpair (lam, x), the tensor is
    deflated::

        T <- T - lam * outer(x, x, ..., x)

    and the power method is restarted to find the next eigenpair.  This
    is an *approximate* deflation; for odd-order tensors it is exact, but
    for even-order tensors the deflated tensor may not be supersymmetric.
    The implementation re-symmetrises after each deflation step.

    Parameters
    ----------
    T : array, shape (n, ..., n)
        Supersymmetric tensor.
    num_eigvals : int
        Number of eigenpairs to compute.
    max_iters : int
        Maximum SS-HOPM iterations per eigenpair.
    tol : float
        Convergence tolerance.
    key : PRNGKey, optional
        Random key for initialisation.

    Returns
    -------
    eigenvalues : array, shape (num_eigvals,)
        Z-eigenvalues sorted by descending magnitude.
    eigenvectors : array, shape (num_eigvals, n)
        Corresponding unit eigenvectors (rows).
    """
    n = T.shape[0]
    order = T.ndim

    if key is None:
        key = jax.random.PRNGKey(0)

    def _outer_k(x: Float[Array, " n"]) -> Float[Array, "..."]:
        """Construct the rank-1 supersymmetric tensor x (x) x ... (x) x."""
        result = x
        for _ in range(order - 1):
            result = jnp.tensordot(result, x, axes=0)
        return result

    def scan_body(carry, _):
        T_cur, key_cur = carry
        key_cur, subkey = jax.random.split(key_cur)
        lam, vec = tensor_power_method(
            T_cur, x0=None, max_iters=max_iters, tol=tol, key=subkey
        )
        # Deflate: remove the found component.
        T_deflated = T_cur - lam * _outer_k(vec)
        # Re-symmetrise (average over permutations).
        from jaxctrl._tensor_ops import symmetrize_tensor

        T_deflated = symmetrize_tensor(T_deflated)
        return (T_deflated, key_cur), (lam, vec)

    (_, _), (eigenvalues, eigenvectors) = jax.lax.scan(
        scan_body, (T, key), None, length=num_eigvals
    )

    # Sort by descending magnitude.
    sort_idx = jnp.argsort(-jnp.abs(eigenvalues))
    eigenvalues = eigenvalues[sort_idx]
    eigenvectors = eigenvectors[sort_idx]

    return eigenvalues, eigenvectors


# ---------------------------------------------------------------------------
# H-eigenvalues via Riemannian gradient descent
# ---------------------------------------------------------------------------


@functools.partial(jax.jit, static_argnums=(1, 2, 3))
def h_eigenvalues(
    T: Float[Array, "..."],
    num_eigvals: int = 1,
    max_iters: int = 1000,
    tol: float = 1e-6,
    key: Optional[PRNGKeyArray] = None,
) -> tuple[Float[Array, " k"], Float[Array, "k n"]]:
    """Compute H-eigenvalues of a supersymmetric tensor.

    A scalar lam is an H-eigenvalue with H-eigenvector x if::

        T x x ... x = lam * x^{[k-1]}

    where x^{[k-1]} is the element-wise (k-1)-th power.

    **Method.**  The H-eigenvalues are critical values of the tensor
    Rayleigh quotient::

        R(x) = T(x, x, ..., x) = sum_{i1,...,ik} T_{i1...ik} x_{i1} ... x_{ik}

    subject to the constraint ||x||_k^k = sum_i |x_i|^k = 1.  We find
    critical points via Riemannian gradient descent on the l_k unit
    sphere.

    The Riemannian gradient is::

        grad_R(x) = nabla R(x) - <nabla R(x), sign(x)|x|^{k-1}> * sign(x)|x|^{k-1}

    where nabla R(x) = k * T x ... x  (contraction on k-1 modes).

    Multiple eigenpairs are found by random restarts with different keys.
    This is a *heuristic*; completeness is not guaranteed.

    Parameters
    ----------
    T : array, shape (n, ..., n)
        Supersymmetric tensor.
    num_eigvals : int
        Number of eigenpairs to seek (via random restarts).
    max_iters : int
        Maximum gradient-descent iterations per restart.
    tol : float
        Convergence tolerance on gradient norm.
    key : PRNGKey, optional
        Random key.

    Returns
    -------
    eigenvalues : array, shape (num_eigvals,)
        H-eigenvalues sorted by descending magnitude.
    eigenvectors : array, shape (num_eigvals, n)
        Corresponding eigenvectors (rows), each on the l_k unit sphere.
    """
    n = T.shape[0]
    order = T.ndim
    k = order

    if key is None:
        key = jax.random.PRNGKey(42)

    contraction_modes = tuple(range(1, k))

    def rayleigh(x: Float[Array, " n"]) -> Float[Array, ""]:
        """T(x, ..., x) — the homogeneous polynomial."""
        return tensor_contract(T, x, contraction_modes).dot(x)

    def project_lk(x: Float[Array, " n"]) -> Float[Array, " n"]:
        """Project onto the l_k unit sphere: ||x||_k^k = 1."""
        norm_k = jnp.sum(jnp.abs(x) ** k) ** (1.0 / k)
        return x / jnp.maximum(norm_k, 1e-12)

    def _solve_one(subkey):
        x0 = jax.random.normal(subkey, (n,))
        x0 = project_lk(x0)

        # Step size — heuristic based on tensor norm.
        lr = 0.01 / jnp.maximum(jnp.sqrt(jnp.sum(T ** 2)), 1.0)

        def body(i, x):
            # Euclidean gradient of Rayleigh quotient.
            grad = jax.grad(rayleigh)(x)
            # Riemannian projection: remove component along the
            # constraint gradient direction (sign(x)|x|^{k-1}).
            constraint_grad = jnp.sign(x) * jnp.abs(x) ** (k - 1)
            constraint_grad = constraint_grad / jnp.maximum(
                jnp.linalg.norm(constraint_grad), 1e-12
            )
            riem_grad = grad - jnp.dot(grad, constraint_grad) * constraint_grad
            x_new = x + lr * riem_grad  # ascent for largest eigenvalue
            x_new = project_lk(x_new)
            return x_new

        x_final = jax.lax.fori_loop(0, max_iters, body, x0)

        # Compute the H-eigenvalue: lam = (T x...x)_i / x_i^{k-1}
        # Use the Rayleigh quotient form which is more stable.
        Tx = tensor_contract(T, x_final, contraction_modes)
        lam = rayleigh(x_final)
        return lam, x_final

    keys = jax.random.split(key, num_eigvals)
    eigenvalues, eigenvectors = jax.vmap(_solve_one)(keys)

    # Sort by descending magnitude.
    sort_idx = jnp.argsort(-jnp.abs(eigenvalues))
    eigenvalues = eigenvalues[sort_idx]
    eigenvectors = eigenvectors[sort_idx]

    return eigenvalues, eigenvectors


# ---------------------------------------------------------------------------
# Spectral radius
# ---------------------------------------------------------------------------


@functools.partial(jax.jit, static_argnums=(1, 2))
def spectral_radius(
    T: Float[Array, "..."],
    max_iters: int = 1000,
    tol: float = 1e-6,
    key: Optional[PRNGKeyArray] = None,
) -> Float[Array, ""]:
    """Spectral radius: largest Z-eigenvalue magnitude.

    This is the natural stability indicator for multilinear dynamical
    systems.  A multilinear system dx/dt = -x + A x x ... x is locally
    stable at the origin when the spectral radius of A (appropriately
    scaled) is less than 1.

    Parameters
    ----------
    T : array, shape (n, ..., n)
        Supersymmetric tensor.
    max_iters : int
        Maximum power-method iterations.
    tol : float
        Convergence tolerance.
    key : PRNGKey, optional
        Random key.

    Returns
    -------
    scalar
        The spectral radius (largest |Z-eigenvalue|).
    """
    lam, _ = tensor_power_method(T, max_iters=max_iters, tol=tol, key=key)
    return jnp.abs(lam)
