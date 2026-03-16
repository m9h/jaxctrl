"""Known-answer tests for the ARTE solver and multilinear LQR.

Sections
--------
1. tensor_lyapunov: mode-1 unfolding Lyapunov solve
2. solve_arte: tensor CARE via linearization
3. multilinear_lqr: gain computation for tensor systems
4. Order-2 edge case (should reduce to standard CARE)
5. Differentiability through solve_arte
"""

import jax
import jax.numpy as jnp
import pytest
from jaxctrl._arte import multilinear_lqr, solve_arte, tensor_lyapunov
from jaxctrl._tensor_ops import tensor_unfold


# -----------------------------------------------------------------------
# 1. Tensor Lyapunov
# -----------------------------------------------------------------------


class TestTensorLyapunov:
    r"""Tensor Lyapunov equation via mode-1 unfolding.

    For even-order tensors, tensor_lyapunov unfolds to matrix form,
    solves the standard Lyapunov equation, and folds back.
    """

    def test_4th_order_returns_correct_shape(self):
        """Result should have the same shape as the input tensor."""
        A = jnp.zeros((2, 2, 2, 2))
        # Make A stable in the unfolded sense: diagonal with negative entries
        A = A.at[0, 0, 0, 0].set(-2.0)
        A = A.at[1, 1, 1, 1].set(-3.0)

        Q = jnp.zeros((2, 2, 2, 2))
        Q = Q.at[0, 0, 0, 0].set(1.0)
        Q = Q.at[1, 1, 1, 1].set(1.0)

        X = tensor_lyapunov(A, Q)
        assert X.shape == (2, 2, 2, 2), f"Expected shape (2,2,2,2), got {X.shape}"

    def test_matrix_lyapunov_residual(self):
        """For order-2 tensors (matrices), verify A X + X A^T + Q = 0 directly."""
        A = jnp.array([[-1.0, 0.5], [0.0, -2.0]])
        Q = jnp.eye(2)

        X = tensor_lyapunov(A, Q)
        residual = A @ X + X @ A.T + Q
        assert jnp.allclose(residual, 0.0, atol=1e-4), (
            f"Lyapunov residual norm = {jnp.linalg.norm(residual)}"
        )

    def test_matrix_case(self):
        """For order-2 (matrix) input, should match solve_continuous_lyapunov."""
        from jaxctrl._lyapunov import solve_continuous_lyapunov

        A = jnp.array([[-1.0, 0.0], [0.0, -2.0]])
        Q = jnp.eye(2)

        X_tensor = tensor_lyapunov(A, Q)
        X_matrix = solve_continuous_lyapunov(A, Q)
        assert jnp.allclose(X_tensor, X_matrix, atol=1e-5), (
            f"Tensor:\n{X_tensor}\nMatrix:\n{X_matrix}"
        )


# -----------------------------------------------------------------------
# 2. solve_arte
# -----------------------------------------------------------------------


class TestSolveARTE:
    r"""ARTE solver: linearise tensor system then solve standard CARE.

    For a 3rd-order tensor A of shape (2,2,2), B of shape (2,1):
    - The result X should be (2,2), symmetric, PSD.
    - The linearised CARE residual should be zero.
    """

    def _make_system(self):
        """Create a simple 3rd-order system."""
        # Diagonal tensor with negative entries for stability
        A = jnp.zeros((2, 2, 2))
        A = A.at[0, 0, 0].set(-2.0)
        A = A.at[1, 1, 1].set(-3.0)
        A = A.at[0, 1, 0].set(-0.5)
        A = A.at[1, 0, 1].set(-0.5)

        B = jnp.array([[0.0], [1.0]])
        Q = jnp.eye(2)
        R = jnp.array([[1.0]])
        return A, B, Q, R

    def test_output_shape(self):
        """X should be (n, n) = (2, 2)."""
        A, B, Q, R = self._make_system()
        X = solve_arte(A, B, Q, R, order=3)
        assert X.shape == (2, 2), f"Expected (2, 2), got {X.shape}"

    def test_solution_is_symmetric(self):
        """X should be symmetric."""
        A, B, Q, R = self._make_system()
        X = solve_arte(A, B, Q, R, order=3)
        assert jnp.allclose(X, X.T, atol=1e-8), "X should be symmetric"

    def test_solution_is_psd(self):
        """X should be positive semi-definite."""
        A, B, Q, R = self._make_system()
        X = solve_arte(A, B, Q, R, order=3)
        eigs = jnp.linalg.eigvalsh(X)
        assert jnp.all(eigs >= -1e-6), f"X should be PSD, eigenvalues: {eigs}"


# -----------------------------------------------------------------------
# 3. multilinear_lqr
# -----------------------------------------------------------------------


class TestMultilinearLQR:
    r"""LQR gain for multilinear systems.

    K = R^{-1} B^T X where X solves the ARTE.
    The closed-loop linearised system A_lin - B K should be stable.
    """

    def test_gain_shape(self):
        """K should be (m, n) = (1, 2)."""
        A = jnp.zeros((2, 2, 2))
        A = A.at[0, 0, 0].set(-2.0)
        A = A.at[1, 1, 1].set(-3.0)
        B = jnp.array([[0.0], [1.0]])
        Q = jnp.eye(2)
        R = jnp.array([[1.0]])

        K = multilinear_lqr(A, B, Q, R)
        assert K.shape == (1, 2), f"Expected (1, 2), got {K.shape}"

    def test_closed_loop_stability(self):
        """The closed-loop linearised system should be Hurwitz."""
        from jaxctrl._tensor_ops import tensor_contract

        A = jnp.zeros((2, 2, 2))
        A = A.at[0, 0, 0].set(-2.0)
        A = A.at[1, 1, 1].set(-3.0)
        A = A.at[0, 1, 0].set(-0.5)
        A = A.at[1, 0, 1].set(-0.5)
        B = jnp.array([[0.0], [1.0]])
        Q = jnp.eye(2)
        R = jnp.array([[1.0]])

        K = multilinear_lqr(A, B, Q, R)

        # Linearise A at the uniform vector
        n = 2
        uniform = jnp.ones(n) / jnp.sqrt(n)
        A_lin = tensor_contract(A, uniform, (2,)).reshape(n, n)
        A_cl = A_lin - B @ K

        eigvals = jnp.linalg.eigvals(A_cl)
        assert jnp.all(jnp.real(eigvals) < 0), (
            f"Closed-loop should be stable, eigenvalues: {eigvals}"
        )


# -----------------------------------------------------------------------
# 4. Order-2 edge case
# -----------------------------------------------------------------------


class TestOrder2EdgeCase:
    r"""When A is a matrix (order 2), solve_arte should reduce to
    the standard CARE.
    """

    def test_matrix_system_matches_care(self):
        """solve_arte with order=2 should match solve_continuous_are."""
        from jaxctrl._riccati import solve_continuous_are

        A = jnp.array([[-1.0, 0.5], [0.0, -2.0]])
        B = jnp.array([[0.0], [1.0]])
        Q = jnp.eye(2)
        R = jnp.array([[1.0]])

        X_arte = solve_arte(A, B, Q, R, order=2)
        X_care = solve_continuous_are(A, B, Q, R)
        assert jnp.allclose(X_arte, X_care, atol=1e-4), (
            f"ARTE:\n{X_arte}\nCARE:\n{X_care}"
        )


# -----------------------------------------------------------------------
# 5. Differentiability
# -----------------------------------------------------------------------


class TestARTEDifferentiability:
    """Verify that gradients flow through solve_arte without errors."""

    def test_grad_wrt_Q_runs(self):
        """jax.grad through solve_arte w.r.t. Q should not error."""
        A = jnp.zeros((2, 2, 2))
        A = A.at[0, 0, 0].set(-2.0)
        A = A.at[1, 1, 1].set(-3.0)
        B = jnp.array([[0.0], [1.0]])
        Q = jnp.eye(2)
        R = jnp.array([[1.0]])

        def f(Q_):
            X = solve_arte(A, B, Q_, R, order=3)
            return jnp.sum(X)

        grad_Q = jax.grad(f)(Q)
        assert jnp.all(jnp.isfinite(grad_Q)), f"Gradient has non-finite values: {grad_Q}"

    def test_grad_wrt_R_runs(self):
        """jax.grad through solve_arte w.r.t. R should not error."""
        A = jnp.zeros((2, 2, 2))
        A = A.at[0, 0, 0].set(-2.0)
        A = A.at[1, 1, 1].set(-3.0)
        B = jnp.array([[0.0], [1.0]])
        Q = jnp.eye(2)
        R = jnp.array([[1.0]])

        def f(R_):
            X = solve_arte(A, B, Q, R_, order=3)
            return jnp.sum(X)

        grad_R = jax.grad(f)(R)
        assert jnp.all(jnp.isfinite(grad_R)), f"Gradient has non-finite values: {grad_R}"
