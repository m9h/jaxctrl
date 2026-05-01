"""Differentiable LQR: gradient of the optimal cost w.r.t. the state weight Q.

The double integrator dx/dt = A x + B u with A = [[0, 1], [0, 0]] and
B = [[0], [1]] is the simplest non-trivial LTI system.  We compute the
LQR cost J(Q) = trace(X(Q)) where X(Q) solves the CARE for given Q, and
take its gradient w.r.t. Q via JAX autodiff.  A finite-difference cross
check confirms the gradient flows correctly through the Schur-method CARE
solver (Kao & Hennequin 2020).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

import jaxctrl

jax.config.update("jax_enable_x64", True)

A = jnp.array([[0.0, 1.0], [0.0, 0.0]])
B = jnp.array([[0.0], [1.0]])
R = jnp.eye(1)


def cost(Q: jax.Array) -> jax.Array:
    """LQR value at the unit initial covariance: J = trace(X)."""
    _, X = jaxctrl.lqr(A, B, Q, R)
    return jnp.trace(X)


def compute_grad() -> jax.Array:
    """Autodiff gradient dJ/dQ at Q = I."""
    Q0 = jnp.eye(2)
    return jax.grad(cost)(Q0)


def _finite_diff(Q0: jax.Array, eps: float = 1e-5) -> jax.Array:
    n = Q0.shape[0]
    fd = jnp.zeros_like(Q0)
    for i in range(n):
        for j in range(n):
            E = jnp.zeros_like(Q0).at[i, j].set(eps)
            fd = fd.at[i, j].set((cost(Q0 + E) - cost(Q0 - E)) / (2.0 * eps))
    return fd


def main() -> None:
    Q0 = jnp.eye(2)
    g = compute_grad()
    fd = _finite_diff(Q0)
    print(f"J(Q=I)             = {float(cost(Q0)):.6f}")
    print(f"autodiff dJ/dQ     =\n{jnp.asarray(g)}")
    print(f"finite-diff dJ/dQ  =\n{jnp.asarray(fd)}")
    print(f"max |ad - fd|      = {float(jnp.max(jnp.abs(g - fd))):.3e}")


if __name__ == "__main__":
    main()
