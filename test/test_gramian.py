"""Known-answer tests for Gramians and controllability analysis.

Every test computes an expected result by hand (documented in the docstring)
and compares it to the implementation output.

Sections
--------
1. Controllability matrix construction and rank
2. Uncontrollable system detection
3. Gramian via Lyapunov equation
4. Minimum control energy
5. Observability (dual) tests
"""

import jax
import jax.numpy as jnp
import pytest
from jaxctrl._gramian import (
    controllability_gramian,
    controllability_matrix,
    observability_gramian,
    observability_matrix,
)
from jaxctrl._controllability import (
    is_controllable,
    is_observable,
    minimum_energy,
)


# -----------------------------------------------------------------------
# 1. Controllability matrix: double integrator
# -----------------------------------------------------------------------


class TestControllabilityMatrix:
    r"""Controllability matrix C = [B, AB, A^2 B, ..., A^{n-1} B].

    Setup
    -----
    A = [[0, 1], [0, 0]],  B = [[0], [1]]

    Hand computation
    ----------------
    AB = [[0,1],[0,0]] @ [[0],[1]] = [[1],[0]]
    C = [B | AB] = [[0, 1],
                     [1, 0]]
    rank(C) = 2 = n  =>  system is controllable.
    """

    def test_controllability_matrix_double_integrator(self, double_integrator):
        """C = [[0, 1], [1, 0]] for the double integrator."""
        A, B = double_integrator
        C = controllability_matrix(A, B)
        expected = jnp.array([[0.0, 1.0], [1.0, 0.0]])
        assert jnp.allclose(C, expected, atol=1e-6), (
            f"Expected:\n{expected}\nGot:\n{C}"
        )

    def test_controllability_rank_is_n(self, double_integrator):
        """The controllability matrix has full rank (= n = 2)."""
        A, B = double_integrator
        C = controllability_matrix(A, B)
        rank = jnp.linalg.matrix_rank(C)
        assert rank == 2, f"Expected rank 2, got {rank}"

    def test_is_controllable_double_integrator(self, double_integrator):
        """Double integrator is controllable."""
        A, B = double_integrator
        assert is_controllable(A, B), "Double integrator should be controllable"

    def test_3x3_chain_integrator(self):
        r"""Triple integrator: x''' = u.

        A = [[0,1,0],[0,0,1],[0,0,0]], B = [[0],[0],[1]]
        AB = [[0],[1],[0]], A^2 B = [[1],[0],[0]]
        C = [[0,0,1],[0,1,0],[1,0,0]]  (antidiagonal identity)
        rank = 3 = n => controllable.
        """
        A = jnp.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
        B = jnp.array([[0.0], [0.0], [1.0]])

        C = controllability_matrix(A, B)
        expected = jnp.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
        assert jnp.allclose(C, expected, atol=1e-6)
        assert is_controllable(A, B)


# -----------------------------------------------------------------------
# 2. Uncontrollable system
# -----------------------------------------------------------------------


class TestUncontrollableSystem:
    r"""A system where one state is unreachable.

    Setup
    -----
    A = [[1, 0], [0, 2]],  B = [[1], [0]]

    Hand computation
    ----------------
    The input only affects state 1.  State 2 is decoupled and unactuated.
    AB = [[1],[0]], so C = [[1, 1], [0, 0]].
    rank(C) = 1 < 2  =>  not controllable.
    """

    def test_uncontrollable_decoupled(self):
        """C = [[1, 1], [0, 0]], rank 1 < n = 2."""
        A = jnp.array([[1.0, 0.0], [0.0, 2.0]])
        B = jnp.array([[1.0], [0.0]])

        C = controllability_matrix(A, B)
        expected = jnp.array([[1.0, 1.0], [0.0, 0.0]])
        assert jnp.allclose(C, expected, atol=1e-6)

    def test_is_not_controllable(self):
        """is_controllable returns False for the decoupled system."""
        A = jnp.array([[1.0, 0.0], [0.0, 2.0]])
        B = jnp.array([[1.0], [0.0]])
        assert not is_controllable(A, B), (
            "Decoupled system with one unactuated state is not controllable"
        )

    def test_rank_deficient(self):
        """rank(C) = 1 for the uncontrollable system."""
        A = jnp.array([[1.0, 0.0], [0.0, 2.0]])
        B = jnp.array([[1.0], [0.0]])
        C = controllability_matrix(A, B)
        rank = jnp.linalg.matrix_rank(C)
        assert rank == 1, f"Expected rank 1, got {rank}"


# -----------------------------------------------------------------------
# 3. Gramian via Lyapunov equation
# -----------------------------------------------------------------------


class TestControllabilityGramian:
    r"""The controllability Gramian W_c satisfies A W_c + W_c A^T + B B^T = 0.

    For a stable A, W_c = solve_continuous_lyapunov(A, B B^T).

    Setup: A = diag(-1, -2), B = [[1], [1]]
    BB^T = [[1, 1], [1, 1]]

    AW + WA^T + BB^T = 0 with diagonal A:
        w11: -2*w11 + 1 = 0  =>  w11 = 0.5
        w22: -4*w22 + 1 = 0  =>  w22 = 0.25
        w12: (-1 + -2)*w12 + 1 = 0  =>  w12 = 1/3
    W_c = [[0.5, 1/3], [1/3, 0.25]]
    """

    def test_gramian_known_answer(self):
        """W_c = [[0.5, 1/3], [1/3, 0.25]] for A=diag(-1,-2), B=[1;1]."""
        A = jnp.array([[-1.0, 0.0], [0.0, -2.0]])
        B = jnp.array([[1.0], [1.0]])

        W = controllability_gramian(A, B)
        expected = jnp.array([[0.5, 1.0 / 3.0], [1.0 / 3.0, 0.25]])
        assert jnp.allclose(W, expected, atol=1e-5), (
            f"Expected:\n{expected}\nGot:\n{W}"
        )

    def test_gramian_satisfies_lyapunov(self):
        """Verify A W + W A^T + B B^T = 0."""
        A = jnp.array([[-1.0, 0.0], [0.0, -2.0]])
        B = jnp.array([[1.0], [1.0]])

        W = controllability_gramian(A, B)
        residual = A @ W + W @ A.T + B @ B.T
        assert jnp.allclose(residual, 0.0, atol=1e-5), (
            f"Lyapunov residual norm = {jnp.linalg.norm(residual)}"
        )

    def test_gramian_is_symmetric_psd(self, key):
        """The controllability Gramian should be symmetric PSD."""
        k1, k2 = jax.random.split(key)
        A_raw = jax.random.normal(k1, (3, 3))
        eigvals = jnp.linalg.eigvals(A_raw)
        shift = jnp.max(jnp.real(eigvals)) + 1.0
        A = A_raw - shift * jnp.eye(3)
        B = jax.random.normal(k2, (3, 2))

        W = controllability_gramian(A, B)
        assert jnp.allclose(W, W.T, atol=1e-8), "Gramian should be symmetric"
        eigs = jnp.linalg.eigvalsh(W)
        assert jnp.all(eigs >= -1e-5), f"Gramian should be PSD, eigs: {eigs}"


# -----------------------------------------------------------------------
# 4. Minimum control energy
# -----------------------------------------------------------------------


class TestMinimumEnergy:
    r"""Minimum energy to steer the double integrator from origin to a target.

    The minimum energy to reach state x_f from the origin in time T is:

        E = x_f^T  W_c(T)^{-1}  x_f

    where W_c(T) = integral_0^T  exp(A t) B B^T exp(A^T t) dt.

    For the double integrator A = [[0,1],[0,0]], B = [[0],[1]], T=1:

        exp(At) = [[1, t], [0, 1]]
        exp(At) B = [[t], [1]]
        B^T exp(A^T t) = [t, 1]

        exp(At) B B^T exp(A^T t) = [[t^2, t], [t, 1]]

        W_c(1) = integral_0^1 [[t^2, t], [t, 1]] dt
               = [[1/3, 1/2], [1/2, 1]]

    To reach x_f = [1, 0]:
        W_c^{-1} = [[12, -6], [-6, 4]]
        E = [1,0] @ [[12,-6],[-6,4]] @ [1,0] = 12.
    """

    def test_minimum_energy_double_integrator(self, double_integrator):
        """Energy to reach [1, 0] from origin in time T=1 is 12."""
        A, B = double_integrator
        x_f = jnp.array([1.0, 0.0])
        T = 1.0

        E = minimum_energy(A, B, jnp.zeros_like(x_f), x_f, T)
        assert jnp.allclose(E, 12.0, atol=0.05), (
            f"Expected energy 12.0, got {E}"
        )

    def test_minimum_energy_easy_direction(self, double_integrator):
        """Energy to reach [0, 1] should be less than [1, 0].

        [0, 1] is the 'directly actuated' direction and requires less energy.
        W_c^{-1} = [[12, -6], [-6, 4]]
        E([0,1]) = [0,1] @ [[12,-6],[-6,4]] @ [0,1] = 4.
        """
        A, B = double_integrator
        x_f = jnp.array([0.0, 1.0])
        T = 1.0

        E = minimum_energy(A, B, jnp.zeros_like(x_f), x_f, T)
        assert jnp.allclose(E, 4.0, atol=0.05), (
            f"Expected energy 4.0, got {E}"
        )


# -----------------------------------------------------------------------
# 5. Observability (dual tests)
# -----------------------------------------------------------------------


class TestObservability:
    r"""Observability is the dual of controllability.

    A system (A, C) is observable iff (A^T, C^T) is controllable.
    The observability matrix is O = [C; CA; CA^2; ...].

    Setup: A = [[0, 1], [0, 0]], C = [[1, 0]] (measure position only)
    O = [C; CA] = [[1, 0], [0, 1]]  =>  rank 2  =>  observable.
    """

    def test_observability_matrix_double_integrator(self):
        """O = [[1, 0], [0, 1]] when measuring position of double integrator."""
        A = jnp.array([[0.0, 1.0], [0.0, 0.0]])
        C = jnp.array([[1.0, 0.0]])

        O = observability_matrix(A, C)
        expected = jnp.array([[1.0, 0.0], [0.0, 1.0]])
        assert jnp.allclose(O, expected, atol=1e-6), (
            f"Expected:\n{expected}\nGot:\n{O}"
        )

    def test_is_observable_double_integrator(self):
        """Double integrator with position measurement is observable."""
        A = jnp.array([[0.0, 1.0], [0.0, 0.0]])
        C = jnp.array([[1.0, 0.0]])
        assert is_observable(A, C), "Position-measured double integrator is observable"

    def test_unobservable_system(self):
        r"""Measuring only velocity of a system with decoupled position.

        A = [[1, 0], [0, 2]], C = [[0, 1]]
        CA = [[0, 2]]
        O = [[0, 1], [0, 2]]  =>  rank 1 < 2  =>  not observable.
        """
        A = jnp.array([[1.0, 0.0], [0.0, 2.0]])
        C = jnp.array([[0.0, 1.0]])

        O = observability_matrix(A, C)
        expected = jnp.array([[0.0, 1.0], [0.0, 2.0]])
        assert jnp.allclose(O, expected, atol=1e-6)
        assert not is_observable(A, C), "Should be unobservable"

    def test_observability_gramian_known_answer(self):
        r"""Observability Gramian W_o satisfies A^T W_o + W_o A + C^T C = 0.

        For A = diag(-1, -2), C = [1, 1]:
        C^T C = [[1, 1], [1, 1]]

        A^T W + W A + C^T C = 0:
            w11: -2*w11 + 1 = 0  =>  w11 = 0.5
            w22: -4*w22 + 1 = 0  =>  w22 = 0.25
            w12: (-1-2)*w12 + 1 = 0  =>  w12 = 1/3

        W_o = [[0.5, 1/3], [1/3, 0.25]]
        (Same structure as controllability Gramian since A is diagonal
        and C^T C = BB^T with B = C^T.)
        """
        A = jnp.array([[-1.0, 0.0], [0.0, -2.0]])
        C = jnp.array([[1.0, 1.0]])

        W = observability_gramian(A, C)
        expected = jnp.array([[0.5, 1.0 / 3.0], [1.0 / 3.0, 0.25]])
        assert jnp.allclose(W, expected, atol=1e-5), (
            f"Expected:\n{expected}\nGot:\n{W}"
        )

    def test_observability_gramian_residual(self):
        """Verify A^T W + W A + C^T C = 0."""
        A = jnp.array([[-1.0, 0.0], [0.0, -2.0]])
        C = jnp.array([[1.0, 1.0]])

        W = observability_gramian(A, C)
        residual = A.T @ W + W @ A + C.T @ C
        assert jnp.allclose(residual, 0.0, atol=1e-5), (
            f"Residual norm = {jnp.linalg.norm(residual)}"
        )
