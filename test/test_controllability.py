"""Known-answer tests for stabilisability, detectability, and minimum energy.

Complements test_gramian.py which covers is_controllable and is_observable.

Sections
--------
1. Stabilisability (PBH test)
2. Detectability (dual PBH test)
3. Minimum energy: scaling and differentiability
"""

import jax
import jax.numpy as jnp
import pytest
from jaxctrl._controllability import (
    is_detectable,
    is_stabilizable,
    minimum_energy,
)


# -----------------------------------------------------------------------
# 1. Stabilisability
# -----------------------------------------------------------------------


class TestStabilisability:
    r"""PBH test for stabilisability.

    (A, B) is stabilisable iff rank([sI - A, B]) = n for every eigenvalue s
    with Re(s) >= 0.  Equivalently, all uncontrollable modes are stable.
    """

    def test_controllable_implies_stabilizable(self):
        """A fully controllable system is always stabilisable.

        Double integrator: A = [[0,1],[0,0]], B = [[0],[1]].
        """
        A = jnp.array([[0.0, 1.0], [0.0, 0.0]])
        B = jnp.array([[0.0], [1.0]])
        assert is_stabilizable(A, B), (
            "Controllable double integrator should be stabilisable"
        )

    def test_uncontrollable_stable_mode_is_stabilizable(self):
        r"""Uncontrollable mode is stable => system is stabilisable.

        A = [[-1, 0], [0, 2]], B = [[0], [1]]
        Eigenvalues: -1 (uncontrollable), 2 (controllable).
        The uncontrollable mode (-1) is stable, so the system is stabilisable.
        """
        A = jnp.array([[-1.0, 0.0], [0.0, 2.0]])
        B = jnp.array([[0.0], [1.0]])
        assert is_stabilizable(A, B), (
            "Uncontrollable stable mode => stabilisable"
        )

    def test_uncontrollable_unstable_mode_not_stabilizable(self):
        r"""Uncontrollable mode is unstable => NOT stabilisable.

        A = [[1, 0], [0, 2]], B = [[0], [1]]
        Eigenvalues: 1 (uncontrollable), 2 (controllable).
        The uncontrollable mode (+1) is unstable, so the system is NOT stabilisable.
        """
        A = jnp.array([[1.0, 0.0], [0.0, 2.0]])
        B = jnp.array([[0.0], [1.0]])
        assert not is_stabilizable(A, B), (
            "Uncontrollable unstable mode => NOT stabilisable"
        )

    def test_stable_system_always_stabilizable(self):
        r"""A Hurwitz matrix is always stabilisable regardless of B.

        A = diag(-1, -2, -3), B = [[1],[0],[0]]
        All eigenvalues are strictly negative, so even though only state 1
        is actuated, the system is stabilisable.
        """
        A = jnp.diag(jnp.array([-1.0, -2.0, -3.0]))
        B = jnp.array([[1.0], [0.0], [0.0]])
        assert is_stabilizable(A, B), (
            "Hurwitz system should be stabilisable with any B"
        )

    def test_complex_eigenvalues(self):
        r"""System with complex conjugate eigenvalues.

        A = [[0, -1], [1, 0]] has eigenvalues ±j (on imaginary axis).
        B = [[1], [0]].
        The controllability matrix C = [B, AB] = [[1, 0], [0, 1]] has rank 2,
        so the system is controllable, hence stabilisable.
        """
        A = jnp.array([[0.0, -1.0], [1.0, 0.0]])
        B = jnp.array([[1.0], [0.0]])
        assert is_stabilizable(A, B), (
            "Controllable system with imaginary eigenvalues is stabilisable"
        )


# -----------------------------------------------------------------------
# 2. Detectability
# -----------------------------------------------------------------------


class TestDetectability:
    r"""Detectability via the dual PBH test.

    (A, C) is detectable iff (A^T, C^T) is stabilisable.
    """

    def test_observable_implies_detectable(self):
        """An observable system is always detectable.

        A = [[0, 1], [0, 0]], C = [[1, 0]] (position measurement).
        O = [[1, 0], [0, 1]] has rank 2 => observable => detectable.
        """
        A = jnp.array([[0.0, 1.0], [0.0, 0.0]])
        C = jnp.array([[1.0, 0.0]])
        assert is_detectable(A, C), "Observable system should be detectable"

    def test_unobservable_stable_mode_is_detectable(self):
        r"""Unobservable mode is stable => detectable.

        A = [[-3, 0], [0, 1]], C = [[0, 1]]
        State 0 (eigenvalue -3) is unobservable but stable.
        State 1 (eigenvalue 1) is observable.
        System is detectable.
        """
        A = jnp.array([[-3.0, 0.0], [0.0, 1.0]])
        C = jnp.array([[0.0, 1.0]])
        assert is_detectable(A, C), (
            "Unobservable stable mode => detectable"
        )

    def test_unobservable_unstable_mode_not_detectable(self):
        r"""Unobservable mode is unstable => NOT detectable.

        A = [[2, 0], [0, -1]], C = [[0, 1]]
        State 0 (eigenvalue 2) is unobservable and unstable.
        System is NOT detectable.
        """
        A = jnp.array([[2.0, 0.0], [0.0, -1.0]])
        C = jnp.array([[0.0, 1.0]])
        assert not is_detectable(A, C), (
            "Unobservable unstable mode => NOT detectable"
        )


# -----------------------------------------------------------------------
# 3. Minimum energy: scaling and differentiability
# -----------------------------------------------------------------------


class TestMinimumEnergyExtended:
    r"""Extended tests for minimum control energy."""

    def test_zero_displacement_from_origin(self):
        """Reaching the origin from the origin requires zero energy."""
        A = jnp.array([[-1.0, 0.0], [0.0, -2.0]])
        B = jnp.array([[1.0], [1.0]])
        x0 = jnp.zeros(2)
        xf = jnp.zeros(2)

        E = minimum_energy(A, B, x0, xf, 1.0)
        assert jnp.allclose(E, 0.0, atol=1e-4), (
            f"Energy should be 0 for x0 == xf == 0, got {E}"
        )

    def test_energy_decreases_with_time(self):
        """More time available => less energy needed.

        This is a fundamental property of controllability Gramians:
        W_c(T1) < W_c(T2) when T1 < T2 (in the PSD ordering sense),
        so the energy E = delta^T W_c^{-1} delta decreases with T.
        """
        A = jnp.array([[0.0, 1.0], [0.0, 0.0]])
        B = jnp.array([[0.0], [1.0]])
        x0 = jnp.zeros(2)
        xf = jnp.array([1.0, 0.0])

        E_short = minimum_energy(A, B, x0, xf, 0.5, num_steps=200)
        E_long = minimum_energy(A, B, x0, xf, 2.0, num_steps=200)

        assert E_short > E_long, (
            f"Energy should decrease with time: E(T=0.5)={E_short}, E(T=2)={E_long}"
        )

    def test_energy_is_differentiable_wrt_T(self):
        """Gradient of energy w.r.t. time horizon T should exist and be negative.

        More time => less energy, so dE/dT < 0.
        """
        A = jnp.array([[0.0, 1.0], [0.0, 0.0]])
        B = jnp.array([[0.0], [1.0]])
        x0 = jnp.zeros(2)
        xf = jnp.array([1.0, 0.0])

        def energy_of_T(T):
            return minimum_energy(A, B, x0, xf, T, num_steps=100)

        T = 1.0
        dE_dT = jax.grad(energy_of_T)(T)
        assert jnp.isfinite(dE_dT), f"dE/dT should be finite, got {dE_dT}"
        assert dE_dT < 0, f"dE/dT should be negative (more time => less energy), got {dE_dT}"

    def test_energy_scales_with_distance(self):
        """Energy should scale quadratically with the displacement magnitude.

        E(alpha * xf) = alpha^2 * E(xf) since E = delta^T W^{-1} delta.
        Uses the double integrator with the same setup as test_gramian.py.
        """
        A = jnp.array([[0.0, 1.0], [0.0, 0.0]])
        B = jnp.array([[0.0], [1.0]])
        x0 = jnp.zeros(2)
        xf = jnp.array([0.0, 1.0])

        E1 = minimum_energy(A, B, x0, xf, 1.0)
        E2 = minimum_energy(A, B, x0, 2.0 * xf, 1.0)

        assert jnp.allclose(E2, 4.0 * E1, rtol=0.05), (
            f"Energy should scale quadratically: E(2x)={E2}, 4*E(x)={4*E1}"
        )
