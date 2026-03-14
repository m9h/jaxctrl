"""Known-answer tests for Lyapunov equation solvers.

Every test computes an expected result by hand (documented in the docstring)
and compares it to the implementation output.  If a test fails, the
implementation has a correctness bug.

Sections
--------
1. Continuous Lyapunov with a 2x2 diagonal stable system
2. Residual verification for a random stable system
3. Discrete Lyapunov with a 2x2 Schur-stable system
4. Gradient through solve_continuous_lyapunov via jax.grad
5. Stability predicates is_stable / is_schur_stable
"""

import jax
import jax.numpy as jnp
import pytest
from jaxctrl._lyapunov import (
    is_schur_stable,
    is_stable,
    solve_continuous_lyapunov,
    solve_discrete_lyapunov,
)


# -----------------------------------------------------------------------
# 1. Continuous Lyapunov: 2x2 diagonal stable system
# -----------------------------------------------------------------------


class TestContinuousLyapunovKnownAnswer:
    r"""Continuous Lyapunov equation A X + X A^T + Q = 0 on a diagonal system.

    Setup
    -----
    A = [[-1,  0],
         [ 0, -2]]     (stable: eigenvalues -1, -2)
    Q = I_2

    Derivation
    ----------
    Since A is diagonal, X is diagonal too.  Writing X = diag(x1, x2):

        A X + X A^T + Q = 0
        => -x1 - x1 + 1 = 0   =>  x1 = 0.5
        => -2*x2 - 2*x2 + 1 = 0  =>  x2 = 0.25

    So X = [[0.5,  0  ],
             [0,   0.25]].
    """

    def test_2x2_diagonal_known_answer(self):
        """X = diag(0.5, 0.25) for A = diag(-1, -2), Q = I."""
        A = jnp.array([[-1.0, 0.0], [0.0, -2.0]])
        Q = jnp.eye(2)

        X = solve_continuous_lyapunov(A, Q)
        expected = jnp.array([[0.5, 0.0], [0.0, 0.25]])
        assert jnp.allclose(X, expected, atol=1e-6), (
            f"Expected:\n{expected}\nGot:\n{X}"
        )

    def test_solution_is_symmetric(self):
        """The solution X must be symmetric for symmetric Q."""
        A = jnp.array([[-1.0, 0.0], [0.0, -2.0]])
        Q = jnp.eye(2)

        X = solve_continuous_lyapunov(A, Q)
        assert jnp.allclose(X, X.T, atol=1e-10), "X should be symmetric"

    def test_3x3_diagonal_known_answer(self):
        r"""Extend to 3x3 diagonal system.

        A = diag(-1, -2, -3), Q = I_3
        => x_k = 1 / (2 * |a_k|)  =>  X = diag(0.5, 0.25, 1/6)
        """
        A = jnp.diag(jnp.array([-1.0, -2.0, -3.0]))
        Q = jnp.eye(3)

        X = solve_continuous_lyapunov(A, Q)
        expected = jnp.diag(jnp.array([0.5, 0.25, 1.0 / 6.0]))
        assert jnp.allclose(X, expected, atol=1e-5)


# -----------------------------------------------------------------------
# 2. Residual verification for a random stable system
# -----------------------------------------------------------------------


class TestContinuousLyapunovResidual:
    """Verify ||A X + X A^T + Q|| < tol for a random Hurwitz matrix."""

    def test_residual_random_stable(self, key):
        r"""Generate a random stable A by shifting eigenvalues to the left.

        Method: Sample random A, compute A_stable = A - (max_real_eig + 1) * I
        so all eigenvalues have Re < -1.  Then solve and check the residual.
        """
        k1, k2 = jax.random.split(key)
        A_raw = jax.random.normal(k1, (4, 4))
        # Make stable: shift eigenvalues left
        eigvals = jnp.linalg.eigvals(A_raw)
        shift = jnp.max(jnp.real(eigvals)) + 1.0
        A = A_raw - shift * jnp.eye(4)

        Q = jax.random.normal(k2, (4, 4))
        Q = Q @ Q.T  # symmetric PSD

        X = solve_continuous_lyapunov(A, Q)
        residual = A @ X + X @ A.T + Q
        assert jnp.allclose(residual, 0.0, atol=1e-4), (
            f"Residual norm = {jnp.linalg.norm(residual)}"
        )

    def test_solution_is_psd_for_psd_q(self, key):
        """When Q is PSD, X should be PSD (all eigenvalues >= 0)."""
        k1, k2 = jax.random.split(key)
        A_raw = jax.random.normal(k1, (3, 3))
        eigvals = jnp.linalg.eigvals(A_raw)
        shift = jnp.max(jnp.real(eigvals)) + 1.0
        A = A_raw - shift * jnp.eye(3)

        Q = jax.random.normal(k2, (3, 3))
        Q = Q @ Q.T

        X = solve_continuous_lyapunov(A, Q)
        eigs_X = jnp.linalg.eigvalsh(X)
        assert jnp.all(eigs_X >= -1e-5), (
            f"X should be PSD, but eigenvalues are {eigs_X}"
        )


# -----------------------------------------------------------------------
# 3. Discrete Lyapunov: 2x2 Schur-stable system
# -----------------------------------------------------------------------


class TestDiscreteLyapunovKnownAnswer:
    r"""Discrete Lyapunov equation A X A^T - X + Q = 0.

    Setup
    -----
    A = [[0.5,  0],
         [0,  0.3]]    (Schur-stable: eigenvalues 0.5, 0.3)
    Q = I_2

    Derivation
    ----------
    Since A is diagonal, X is diagonal.  Writing X = diag(x1, x2):

        0.25 * x1 - x1 + 1 = 0   =>  x1 = 1 / 0.75 = 4/3
        0.09 * x2 - x2 + 1 = 0   =>  x2 = 1 / 0.91 = 100/91

    So X = diag(4/3, 100/91).
    """

    def test_2x2_diagonal_known_answer(self):
        """X = diag(4/3, 100/91) for A = diag(0.5, 0.3), Q = I."""
        A = jnp.array([[0.5, 0.0], [0.0, 0.3]])
        Q = jnp.eye(2)

        X = solve_discrete_lyapunov(A, Q)
        expected = jnp.array([[4.0 / 3.0, 0.0], [0.0, 100.0 / 91.0]])
        assert jnp.allclose(X, expected, atol=1e-5), (
            f"Expected:\n{expected}\nGot:\n{X}"
        )

    def test_discrete_residual(self):
        """Verify A X A^T - X + Q = 0 for the known-answer system."""
        A = jnp.array([[0.5, 0.0], [0.0, 0.3]])
        Q = jnp.eye(2)

        X = solve_discrete_lyapunov(A, Q)
        residual = A @ X @ A.T - X + Q
        assert jnp.allclose(residual, 0.0, atol=1e-5), (
            f"Residual norm = {jnp.linalg.norm(residual)}"
        )

    def test_discrete_residual_random(self, key):
        """Verify A X A^T - X + Q = 0 for a random Schur-stable A."""
        k1, k2 = jax.random.split(key)
        A_raw = jax.random.normal(k1, (4, 4))
        # Make Schur-stable: scale so spectral radius < 1
        eigvals = jnp.linalg.eigvals(A_raw)
        spectral_radius = jnp.max(jnp.abs(eigvals))
        A = A_raw / (spectral_radius + 1.0)

        Q = jax.random.normal(k2, (4, 4))
        Q = Q @ Q.T

        X = solve_discrete_lyapunov(A, Q)
        residual = A @ X @ A.T - X + Q
        assert jnp.allclose(residual, 0.0, atol=1e-4), (
            f"Residual norm = {jnp.linalg.norm(residual)}"
        )


# -----------------------------------------------------------------------
# 4. Gradient through solve_continuous_lyapunov
# -----------------------------------------------------------------------


class TestContinuousLyapunovGradient:
    r"""Differentiate through the Lyapunov solver via the custom VJP.

    We define f(A) = sum(solve_continuous_lyapunov(A, Q)) and compare
    jax.grad(f) against a central finite-difference approximation.
    """

    def test_grad_wrt_A_finite_differences(self):
        r"""Gradient of tr(X) w.r.t. A, verified by finite differences.

        For A = diag(-1, -2), Q = I:
            X = diag(0.5, 0.25)
            f(A) = sum(X) = 0.75

        We perturb each entry of A by eps and compute (f(A+eps) - f(A-eps)) / 2eps.
        """
        A = jnp.array([[-1.0, 0.0], [0.0, -2.0]])
        Q = jnp.eye(2)

        def f(A_):
            X = solve_continuous_lyapunov(A_, Q)
            return jnp.sum(X)

        grad_auto = jax.grad(f)(A)

        # Finite differences
        eps = 1e-5
        grad_fd = jnp.zeros_like(A)
        for i in range(2):
            for j in range(2):
                e_ij = jnp.zeros((2, 2)).at[i, j].set(eps)
                fp = f(A + e_ij)
                fm = f(A - e_ij)
                grad_fd = grad_fd.at[i, j].set((fp - fm) / (2.0 * eps))

        assert jnp.allclose(grad_auto, grad_fd, atol=1e-3), (
            f"Auto grad:\n{grad_auto}\nFD grad:\n{grad_fd}"
        )

    def test_grad_wrt_Q(self):
        r"""Gradient of sum(X) w.r.t. Q.

        For diagonal A = diag(-1, -2), the adjoint equation gives
        dL/dQ = -S where A^T S + S A + G_sym = 0 with G = ones(2,2).

        We verify by finite differences.
        """
        A = jnp.array([[-1.0, 0.0], [0.0, -2.0]])
        Q = jnp.eye(2)

        def f(Q_):
            X = solve_continuous_lyapunov(A, Q_)
            return jnp.sum(X)

        grad_auto = jax.grad(f)(Q)

        eps = 1e-5
        grad_fd = jnp.zeros_like(Q)
        for i in range(2):
            for j in range(2):
                e_ij = jnp.zeros((2, 2)).at[i, j].set(eps)
                fp = f(Q + e_ij)
                fm = f(Q - e_ij)
                grad_fd = grad_fd.at[i, j].set((fp - fm) / (2.0 * eps))

        assert jnp.allclose(grad_auto, grad_fd, atol=1e-3), (
            f"Auto grad:\n{grad_auto}\nFD grad:\n{grad_fd}"
        )

    def test_grad_discrete_wrt_A_finite_differences(self):
        """Gradient of sum(X) w.r.t. A for the discrete Lyapunov, vs FD."""
        A = jnp.array([[0.5, 0.0], [0.0, 0.3]])
        Q = jnp.eye(2)

        def f(A_):
            X = solve_discrete_lyapunov(A_, Q)
            return jnp.sum(X)

        grad_auto = jax.grad(f)(A)

        eps = 1e-5
        grad_fd = jnp.zeros_like(A)
        for i in range(2):
            for j in range(2):
                e_ij = jnp.zeros((2, 2)).at[i, j].set(eps)
                fp = f(A + e_ij)
                fm = f(A - e_ij)
                grad_fd = grad_fd.at[i, j].set((fp - fm) / (2.0 * eps))

        assert jnp.allclose(grad_auto, grad_fd, atol=0.02), (
            f"Auto grad:\n{grad_auto}\nFD grad:\n{grad_fd}"
        )


# -----------------------------------------------------------------------
# 5. Stability predicates
# -----------------------------------------------------------------------


class TestStabilityPredicates:
    """Test is_stable (continuous) and is_schur_stable (discrete)."""

    def test_is_stable_hurwitz(self):
        """A = diag(-1, -2) is Hurwitz (all eigenvalues in open left half-plane)."""
        A = jnp.array([[-1.0, 0.0], [0.0, -2.0]])
        assert is_stable(A), "diag(-1, -2) should be stable"

    def test_is_stable_unstable(self):
        """A = diag(1, -2) has eigenvalue +1 in the right half-plane."""
        A = jnp.array([[1.0, 0.0], [0.0, -2.0]])
        assert not is_stable(A), "diag(1, -2) should be unstable"

    def test_is_stable_marginal(self):
        """A = diag(0, -1) has eigenvalue 0 on the imaginary axis.

        This is marginally stable, but is_stable requires strict Re < 0.
        """
        A = jnp.array([[0.0, 0.0], [0.0, -1.0]])
        assert not is_stable(A), "diag(0, -1) is not strictly stable"

    def test_is_stable_complex_eigenvalues(self):
        """A = [[0, -1], [1, 0]] has eigenvalues +/- j (imaginary axis).

        Not strictly stable.
        """
        A = jnp.array([[0.0, -1.0], [1.0, 0.0]])
        assert not is_stable(A), "purely imaginary eigenvalues are not stable"

    def test_is_stable_damped_oscillator(self):
        """A = [[-0.1, -1], [1, -0.1]] has eigenvalues -0.1 +/- j.

        Strictly stable (damped oscillation).
        """
        A = jnp.array([[-0.1, -1.0], [1.0, -0.1]])
        assert is_stable(A), "damped oscillator should be stable"

    def test_is_schur_stable_inside_disk(self):
        """A = diag(0.5, 0.3) has spectral radius 0.5 < 1."""
        A = jnp.array([[0.5, 0.0], [0.0, 0.3]])
        assert is_schur_stable(A), "diag(0.5, 0.3) should be Schur-stable"

    def test_is_schur_stable_outside_disk(self):
        """A = diag(1.5, 0.3) has eigenvalue 1.5 outside the unit disk."""
        A = jnp.array([[1.5, 0.0], [0.0, 0.3]])
        assert not is_schur_stable(A), "diag(1.5, 0.3) is not Schur-stable"

    def test_is_schur_stable_on_boundary(self):
        """A = diag(1.0, 0.5) has eigenvalue 1.0 on the unit circle.

        Not strictly Schur-stable.
        """
        A = jnp.array([[1.0, 0.0], [0.0, 0.5]])
        assert not is_schur_stable(A), "eigenvalue on unit circle is not Schur-stable"
