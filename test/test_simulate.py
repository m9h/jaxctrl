"""Tests for ODE simulation with Diffrax and fallback.

Sections
--------
1. Open-loop simulation of a stable system
2. Closed-loop LQR simulation
3. Diffrax vs matrix-exponential agreement
4. Callable vs array control input
"""

import jax
import jax.numpy as jnp
import pytest
from jaxctrl._simulate import simulate_closed_loop, simulate_lti, HAS_DIFFRAX


# -----------------------------------------------------------------------
# 1. Open-loop simulation
# -----------------------------------------------------------------------


class TestSimulateLTI:
    r"""Open-loop simulation of dx/dt = Ax + Bu.

    Setup: A = [[-1, 0], [0, -2]], B = [[1], [0]], u = 0.
    Solution: x(t) = [x0_0 * e^{-t}, x0_1 * e^{-2t}].
    """

    def test_free_response_decays(self):
        """A stable system with zero input should decay to the origin."""
        A = jnp.array([[-1.0, 0.0], [0.0, -2.0]])
        B = jnp.array([[1.0], [0.0]])
        x0 = jnp.array([1.0, 1.0])

        def u_zero(t, x):
            return jnp.zeros(1)

        ts, xs = simulate_lti(A, B, x0, u_zero, T=5.0, num_steps=100)

        assert xs.shape == (101, 2)
        assert ts.shape == (101,)
        # Should decay to near zero
        assert jnp.linalg.norm(xs[-1]) < 0.05, (
            f"Final state should be near zero, got {xs[-1]}"
        )

    def test_free_response_analytical(self):
        """Compare numerical solution to analytical for a diagonal system."""
        A = jnp.array([[-1.0, 0.0], [0.0, -2.0]])
        B = jnp.array([[1.0], [0.0]])
        x0 = jnp.array([1.0, 1.0])

        def u_zero(t, x):
            return jnp.zeros(1)

        ts, xs = simulate_lti(A, B, x0, u_zero, T=3.0, num_steps=200)

        # Analytical: x(t) = [e^{-t}, e^{-2t}]
        x_analytical = jnp.column_stack([
            jnp.exp(-ts),
            jnp.exp(-2 * ts),
        ])
        assert jnp.allclose(xs, x_analytical, atol=0.01), (
            f"Max error: {jnp.max(jnp.abs(xs - x_analytical))}"
        )

    def test_constant_input_steady_state(self):
        """Constant input u=1 on stable system should reach steady state.

        Steady state: 0 = A x_ss + B u => x_ss = -A^{-1} B u.
        For A = diag(-1, -2), B = [[1], [1]], u = [1]:
            x_ss = [1, 0.5].
        """
        A = jnp.array([[-1.0, 0.0], [0.0, -2.0]])
        B = jnp.array([[1.0], [1.0]])
        x0 = jnp.zeros(2)

        def u_const(t, x):
            return jnp.ones(1)

        ts, xs = simulate_lti(A, B, x0, u_const, T=10.0, num_steps=200)

        x_ss = jnp.array([1.0, 0.5])
        assert jnp.allclose(xs[-1], x_ss, atol=0.02), (
            f"Expected steady state {x_ss}, got {xs[-1]}"
        )

    def test_array_input(self):
        """Accept a (num_steps, m) array as control schedule."""
        A = jnp.array([[-1.0, 0.0], [0.0, -2.0]])
        B = jnp.array([[1.0], [0.0]])
        x0 = jnp.zeros(2)

        num_steps = 50
        u_schedule = jnp.ones((num_steps, 1))
        ts, xs = simulate_lti(A, B, x0, u_schedule, T=5.0, num_steps=num_steps)

        assert xs.shape == (num_steps + 1, 2)
        # With constant input 1 on first state, should approach x1_ss = 1
        assert xs[-1, 0] > 0.5, f"First state should grow, got {xs[-1, 0]}"


# -----------------------------------------------------------------------
# 2. Closed-loop LQR simulation
# -----------------------------------------------------------------------


class TestSimulateClosedLoop:
    r"""Closed-loop simulation with LQR feedback u = -K x.

    Setup: double integrator with LQR gain.
    """

    def test_closed_loop_stabilises(self):
        """LQR controller should drive the state to the origin."""
        from jaxctrl._riccati import lqr

        A = jnp.array([[0.0, 1.0], [0.0, 0.0]])
        B = jnp.array([[0.0], [1.0]])
        Q = jnp.eye(2)
        R = jnp.array([[1.0]])

        K, _ = lqr(A, B, Q, R)

        x0 = jnp.array([2.0, 1.0])
        ts, xs, us = simulate_closed_loop(A, B, K, x0, T=10.0, num_steps=200)

        assert xs.shape == (201, 2)
        assert us.shape == (201, 1)
        # Should converge to origin
        assert jnp.linalg.norm(xs[-1]) < 0.1, (
            f"State should converge to origin, got {xs[-1]}"
        )

    def test_reference_moves_toward_target(self):
        """Closed-loop with nonzero reference should move toward the reference.

        Note: LQR with u = -K(x-ref) does not achieve zero steady-state
        error for systems with nonzero A (would need integral action).
        We check that the final state is closer to ref than x0.
        """
        from jaxctrl._riccati import lqr

        A = jnp.array([[-1.0, 0.0], [0.0, -2.0]])
        B = jnp.eye(2)
        Q = 10.0 * jnp.eye(2)
        R = jnp.eye(2)

        K, _ = lqr(A, B, Q, R)
        x0 = jnp.zeros(2)
        ref = jnp.array([1.0, 0.5])

        ts, xs, us = simulate_closed_loop(
            A, B, K, x0, T=10.0, num_steps=200, reference=ref
        )

        dist_initial = jnp.linalg.norm(x0 - ref)
        dist_final = jnp.linalg.norm(xs[-1] - ref)
        assert dist_final < dist_initial, (
            f"State should move toward reference: "
            f"initial dist={dist_initial}, final dist={dist_final}"
        )


# -----------------------------------------------------------------------
# 3. Diffrax vs fallback agreement
# -----------------------------------------------------------------------


class TestSolverAgreement:
    """Diffrax and matrix-exponential solvers should agree."""

    @pytest.mark.skipif(not HAS_DIFFRAX, reason="diffrax not installed")
    def test_diffrax_vs_expm(self):
        """Both solvers should produce similar trajectories."""
        A = jnp.array([[-1.0, 0.5], [-0.5, -2.0]])
        B = jnp.array([[1.0], [0.0]])
        x0 = jnp.array([1.0, -0.5])

        def u_fn(t, x):
            return jnp.array([jnp.sin(t)])

        _, xs_diffrax = simulate_lti(
            A, B, x0, u_fn, T=5.0, num_steps=200, use_diffrax=True
        )
        _, xs_expm = simulate_lti(
            A, B, x0, u_fn, T=5.0, num_steps=200, use_diffrax=False
        )

        assert jnp.allclose(xs_diffrax, xs_expm, atol=0.05), (
            f"Max disagreement: {jnp.max(jnp.abs(xs_diffrax - xs_expm))}"
        )

    def test_fallback_works_without_diffrax(self):
        """The matrix-exponential fallback should always work."""
        A = jnp.array([[-1.0, 0.0], [0.0, -2.0]])
        B = jnp.array([[1.0], [0.0]])
        x0 = jnp.array([1.0, 1.0])

        def u_zero(t, x):
            return jnp.zeros(1)

        ts, xs = simulate_lti(
            A, B, x0, u_zero, T=3.0, num_steps=100, use_diffrax=False
        )
        assert xs.shape == (101, 2)
        assert jnp.all(jnp.isfinite(xs))


# -----------------------------------------------------------------------
# 4. Differentiability
# -----------------------------------------------------------------------


class TestSimulationDifferentiability:
    """Simulation should be differentiable w.r.t. initial conditions."""

    def test_grad_wrt_x0(self):
        """Gradient of final state w.r.t. initial state should exist."""
        A = jnp.array([[-1.0, 0.0], [0.0, -2.0]])
        B = jnp.array([[1.0], [0.0]])

        def u_zero(t, x):
            return jnp.zeros(1)

        def final_state_norm(x0):
            _, xs = simulate_lti(
                A, B, x0, u_zero, T=2.0, num_steps=50, use_diffrax=False
            )
            return jnp.sum(xs[-1] ** 2)

        x0 = jnp.array([1.0, 1.0])
        grad = jax.grad(final_state_norm)(x0)
        assert jnp.all(jnp.isfinite(grad)), f"Gradient should be finite: {grad}"
