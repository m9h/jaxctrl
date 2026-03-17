# Copyright 2024 jaxctrl contributors. Apache-2.0 license.
"""ODE simulation for linear and controlled dynamical systems.

Provides Diffrax-based adaptive ODE solvers for simulating LTI systems
with optional feedback control.  Falls back to matrix-exponential
discretisation when Diffrax is not installed.

Usage::

    import jaxctrl
    A = ...  # system matrix
    B = ...  # input matrix
    K, P = jaxctrl.lqr(A, B, Q, R)
    traj = jaxctrl.simulate_closed_loop(A, B, K, x0, T=10.0)
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from jax import lax

try:
    import diffrax

    HAS_DIFFRAX = True
except ImportError:
    diffrax = None  # type: ignore[assignment]
    HAS_DIFFRAX = False


# ===================================================================
# simulate_lti: open-loop LTI simulation
# ===================================================================


def simulate_lti(
    A: jax.Array,
    B: jax.Array,
    x0: jax.Array,
    u_fn: Union[Callable[[float, jax.Array], jax.Array], jax.Array],
    T: float,
    num_steps: int = 200,
    *,
    use_diffrax: bool = True,
) -> Tuple[jax.Array, jax.Array]:
    """Simulate the LTI system dx/dt = A x + B u(t, x).

    Parameters
    ----------
    A : (n, n) array
        System matrix.
    B : (n, m) array
        Input matrix.
    x0 : (n,) array
        Initial state.
    u_fn : callable or (num_steps, m) array
        Control input.  If callable, ``u_fn(t, x)`` returns the control
        at time *t* and state *x*.  If an array, it is treated as a
        piecewise-constant schedule over *num_steps* intervals.
    T : float
        Simulation horizon.
    num_steps : int
        Number of time steps (for saving and for the fallback solver).
    use_diffrax : bool
        If True (default) and Diffrax is installed, use an adaptive
        Dormand-Prince solver.  Otherwise use matrix-exponential
        discretisation.

    Returns
    -------
    ts : (num_steps + 1,) array
        Time points.
    xs : (num_steps + 1, n) array
        State trajectory.
    """
    # Convert array schedule to a callable
    if not callable(u_fn):
        u_schedule = jnp.asarray(u_fn)
        dt = T / u_schedule.shape[0]

        def u_fn_from_schedule(t, x):
            idx = jnp.clip(jnp.int32(t / dt), 0, u_schedule.shape[0] - 1)
            return u_schedule[idx]

        u_fn = u_fn_from_schedule

    if HAS_DIFFRAX and use_diffrax:
        return _simulate_diffrax(A, B, x0, u_fn, T, num_steps)
    else:
        return _simulate_expm(A, B, x0, u_fn, T, num_steps)


def _simulate_diffrax(A, B, x0, u_fn, T, num_steps):
    """Adaptive ODE simulation using Diffrax's Dormand-Prince solver."""

    def vector_field(t, x, args):
        u = u_fn(t, x)
        return A @ x + B @ u

    term = diffrax.ODETerm(vector_field)
    solver = diffrax.Dopri5()
    saveat = diffrax.SaveAt(ts=jnp.linspace(0.0, T, num_steps + 1))
    stepsize_controller = diffrax.PIDController(rtol=1e-6, atol=1e-8)

    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=0.0,
        t1=T,
        dt0=T / num_steps,
        y0=x0,
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        max_steps=num_steps * 16,
    )

    return sol.ts, sol.ys


def _simulate_expm(A, B, x0, u_fn, T, num_steps):
    """Matrix-exponential fallback (zero-order hold discretisation)."""
    n = A.shape[0]
    dt = T / num_steps
    I_n = jnp.eye(n, dtype=A.dtype)
    eAdt = jax.scipy.linalg.expm(A * dt)
    # B_d = A^{-1}(e^{Adt} - I) B; use solve for stability
    B_d = jnp.linalg.solve(A + 1e-12 * I_n, (eAdt - I_n) @ B)

    ts = jnp.linspace(0.0, T, num_steps + 1)

    def step_fn(x, t):
        u = u_fn(t, x)
        x_next = eAdt @ x + B_d @ u
        return x_next, x_next

    _, trajectory = lax.scan(step_fn, x0, ts[:-1])
    trajectory = jnp.concatenate([x0[None, :], trajectory], axis=0)
    return ts, trajectory


# ===================================================================
# simulate_closed_loop: LQR feedback simulation
# ===================================================================


def simulate_closed_loop(
    A: jax.Array,
    B: jax.Array,
    K: jax.Array,
    x0: jax.Array,
    T: float,
    num_steps: int = 200,
    *,
    reference: Optional[jax.Array] = None,
    use_diffrax: bool = True,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """Simulate dx/dt = A x + B u with feedback u = -K (x - ref).

    Parameters
    ----------
    A : (n, n) array
        System matrix.
    B : (n, m) array
        Input matrix.
    K : (m, n) array
        Feedback gain (e.g., from :func:`jaxctrl.lqr`).
    x0 : (n,) array
        Initial state.
    T : float
        Simulation horizon.
    num_steps : int
        Number of time steps.
    reference : (n,) array, optional
        Reference state for tracking.  Default is the origin.
    use_diffrax : bool
        Whether to use Diffrax (default True).

    Returns
    -------
    ts : (num_steps + 1,) array
        Time points.
    xs : (num_steps + 1, n) array
        State trajectory.
    us : (num_steps + 1, m) array
        Control trajectory.
    """
    if reference is None:
        reference = jnp.zeros_like(x0)

    def u_fn(t, x):
        return -K @ (x - reference)

    ts, xs = simulate_lti(A, B, x0, u_fn, T, num_steps, use_diffrax=use_diffrax)

    # Reconstruct control trajectory
    us = jax.vmap(lambda x: -K @ (x - reference))(xs)

    return ts, xs, us
