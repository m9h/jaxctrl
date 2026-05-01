"""Multilinear (tensor) LQR via the matricization approach to ARTE.

We build a small order-3 system tensor A of shape (n, n, n) symmetric in
its last two modes, an input matrix B, and quadratic costs Q, R.  The
matricization-based ARTE solver (Wang & Wei 2024) returns a CARE-style
solution X; multilinear_lqr returns the corresponding feedback gain K.
We report the spectral radius of (A_lin - B K) where A_lin is the
contraction of A used internally by the solver.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

import jaxctrl

jax.config.update("jax_enable_x64", True)

N = 3
M = 1


def _build_system() -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    key = jax.random.PRNGKey(0)
    A_raw = jax.random.normal(key, (N, N, N))
    A = jaxctrl.symmetrize_tensor(A_raw)
    B = jnp.zeros((N, M)).at[-1, 0].set(1.0)
    Q = jnp.eye(N)
    R = jnp.eye(M)
    return A, B, Q, R


def main() -> None:
    A, B, Q, R = _build_system()
    X = jaxctrl.solve_arte(A, B, Q, R, order=3, refine=False)
    K = jaxctrl.multilinear_lqr(A, B, Q, R)

    uniform = jnp.ones(N) / jnp.sqrt(N)
    A_lin = jaxctrl.tensor_contract(A, uniform, (2,)).reshape(N, N)
    closed_loop = A_lin - B @ K
    spec = float(jnp.max(jnp.abs(jnp.linalg.eigvals(closed_loop))))

    print(f"order-3 ARTE solution X shape = {X.shape}")
    print(f"trace(X)                       = {float(jnp.trace(X)):.6f}")
    print(f"K shape                        = {K.shape}")
    print(f"|max eig(A_lin - B K)|         = {spec:.6f}")


if __name__ == "__main__":
    main()
