"""Cross-check CARE and DARE custom_vjp gradients against finite differences.

The existing test_riccati.py covers only the Q-gradient.  This file adds
comprehensive checks against the *scalar* LQR cost J = trace(X) so that
gradients w.r.t. A, B, Q, R can each be cross-checked element-wise against
central finite differences.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

import jaxctrl


def _fd_grad(f, x, eps=1e-5):
    g = jnp.zeros_like(x)
    n = x.size
    for i in range(n):
        ei = jnp.zeros(n).at[i].set(eps).reshape(x.shape)
        g = g.at[tuple(jnp.unravel_index(i, x.shape))].set(
            (f(x + ei) - f(x - ei)) / (2.0 * eps)
        )
    return g


# ----------------------------------------------------------------------
# CARE / continuous LQR
# ----------------------------------------------------------------------


@pytest.fixture
def care_system():
    A = jnp.array([[0.0, 1.0], [0.0, 0.0]])
    B = jnp.array([[0.0], [1.0]])
    Q = jnp.eye(2)
    R = jnp.eye(1)
    return A, B, Q, R


def _care_cost(A, B, Q, R):
    _, X = jaxctrl.lqr(A, B, Q, R)
    return jnp.trace(X)


@pytest.mark.parametrize("param_name", ["A", "B", "Q", "R"])
def test_care_grad_matches_finite_diff(care_system, param_name):
    A, B, Q, R = care_system
    params = {"A": A, "B": B, "Q": Q, "R": R}
    target = params[param_name]

    def f(x):
        local = dict(params)
        local[param_name] = x
        return _care_cost(local["A"], local["B"], local["Q"], local["R"])

    g_ad = jax.grad(f)(target)
    g_fd = _fd_grad(f, target)

    # Q-gradient is canonically the symmetric part: solve_continuous_are
    # depends only on (Q + Q^T)/2, so antisymmetric FD perturbations are
    # numerical noise.  Compare symmetric parts for Q.
    if param_name == "Q":
        g_ad = (g_ad + g_ad.T) / 2.0
        g_fd = (g_fd + g_fd.T) / 2.0

    assert jnp.allclose(g_ad, g_fd, atol=1e-4), (
        f"CARE d{param_name}: ad={g_ad.tolist()}, fd={g_fd.tolist()}"
    )


# ----------------------------------------------------------------------
# DARE / discrete LQR
# ----------------------------------------------------------------------


@pytest.fixture
def dare_system():
    A = jnp.array([[1.0, 0.1], [0.0, 1.0]])
    B = jnp.array([[0.005], [0.1]])
    Q = jnp.eye(2)
    R = jnp.eye(1)
    return A, B, Q, R


def _dare_cost(A, B, Q, R):
    _, X = jaxctrl.dlqr(A, B, Q, R)
    return jnp.trace(X)


@pytest.mark.parametrize("param_name", ["A", "B", "Q", "R"])
def test_dare_grad_matches_finite_diff(dare_system, param_name):
    A, B, Q, R = dare_system
    params = {"A": A, "B": B, "Q": Q, "R": R}
    target = params[param_name]

    def f(x):
        local = dict(params)
        local[param_name] = x
        return _dare_cost(local["A"], local["B"], local["Q"], local["R"])

    g_ad = jax.grad(f)(target)
    g_fd = _fd_grad(f, target)

    # Like CARE, the DARE depends only on the symmetric part of Q; compare
    # symmetric components only.
    if param_name == "Q":
        g_ad = (g_ad + g_ad.T) / 2.0
        g_fd = (g_fd + g_fd.T) / 2.0

    assert jnp.allclose(g_ad, g_fd, atol=1e-3), (
        f"DARE d{param_name}: ad={g_ad.tolist()}, fd={g_fd.tolist()}"
    )
