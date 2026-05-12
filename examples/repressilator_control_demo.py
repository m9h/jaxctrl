"""Controlling the repressilator — a worked cellular-systems example.

The repressilator (Elowitz & Leibler, *Nature* 2000) is the canonical synthetic
gene-regulatory oscillator: three genes wired in a repression ring (1 ⊣ 2 ⊣ 3 ⊣ 1).
In dimensionless form (mRNA m_i, protein p_i, i = 1..3, with gene i repressed by
protein i-1 mod 3):

    dm_i/dt = -m_i + alpha / (1 + p_{i-1}^n) + alpha0
    dp_i/dt = -beta * (p_i - m_i)

This is a tiny GRN with a *known* unstable interior fixed point, so it is an
ideal first demo of the "identify a model → do control theory on it" pipeline
that jaxctrl is built for (it also mirrors ``diff_lqr_demo.py`` but on a
biological system).  We:

  1. derive the symmetric interior fixed point x* (solve p* = alpha/(1+p*^n) + alpha0),
  2. linearise the 6-D dynamics around x* (jax.jacfwd),
  3. test controllability of the linearised GRN from a single actuator
     (a "drug" that adds/removes mRNA of gene 1) -- jaxctrl.is_controllable,
  4. design an LQR feedback law u = -K (x - x*) that quenches the oscillation
     -- jaxctrl.lqr -- and simulate the closed loop -- jaxctrl.simulate_closed_loop,
  5. differentiate the LQR cost J(n) = trace(X(n)) w.r.t. the Hill coefficient n
     via jax.grad (parameter-sensitivity of controllability/control cost),
     with a finite-difference cross-check.

Run:  python examples/repressilator_control_demo.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

import jaxctrl

jax.config.update("jax_enable_x64", True)

# --- kinetic parameters (Elowitz & Leibler-ish, dimensionless) ----------------
ALPHA = 50.0     # max transcription rate (in units of mRNA decay)
ALPHA0 = 0.5     # leaky transcription
BETA = 2.0       # protein-to-mRNA decay ratio
HILL_N = 2.0     # Hill coefficient (cooperativity of repression)

PERM = jnp.array([2, 0, 1])  # gene i is repressed by protein PERM[i] (i-1 mod 3)


def repressilator_rhs(x: jax.Array, n: float = HILL_N) -> jax.Array:
    """dx/dt for x = [m1, m2, m3, p1, p2, p3]."""
    m, p = x[:3], x[3:]
    dm = -m + ALPHA / (1.0 + p[PERM] ** n) + ALPHA0
    dp = -BETA * (p - m)
    return jnp.concatenate([dm, dp])


def fixed_point(n: float = HILL_N) -> jax.Array:
    """Symmetric interior fixed point: m_i = p_i = p*, p* = alpha/(1+p*^n)+alpha0."""
    def g(ps):  # fixed-point map for p*
        return ALPHA / (1.0 + ps ** n) + ALPHA0
    ps = 1.0
    for _ in range(200):  # simple fixed-point iteration (the map is a contraction here)
        ps = 0.5 * ps + 0.5 * g(ps)
    return jnp.full(6, ps)


def linearise(n: float = HILL_N):
    """Jacobian A = ∂f/∂x at the interior fixed point."""
    xs = fixed_point(n)
    A = jax.jacfwd(lambda x: repressilator_rhs(x, n))(xs)
    return A, xs


# --- single-actuator input matrix: a "drug" acting on m1 ----------------------
B = jnp.zeros((6, 1)).at[0, 0].set(1.0)
Q = jnp.eye(6)
R = jnp.eye(1)


def lqr_cost(n: float) -> jax.Array:
    """LQR value J(n) = trace(X(n)) where X solves the CARE for the linearised GRN."""
    A, _ = linearise(n)
    _, X = jaxctrl.lqr(A, B, Q, R)
    return jnp.trace(X)


def _finite_diff(f, x0: float, eps: float = 1e-4) -> float:
    return float((f(x0 + eps) - f(x0 - eps)) / (2.0 * eps))


def main() -> None:
    A, xs = linearise()
    eig = jnp.linalg.eigvals(A)
    n_unstable = int(jnp.sum(eig.real > 0))
    print(f"interior fixed point  x* = {float(xs[0]):.4f} (all coords, by symmetry)")
    print(f"open-loop eigenvalues : {n_unstable}/6 with Re(lambda) > 0  -> oscillatory")

    Cmat = jaxctrl.controllability_matrix(A, B)
    rank = int(jnp.linalg.matrix_rank(Cmat))
    ctrl = bool(jaxctrl.is_controllable(A, B))
    print(f"controllable from a single m1 actuator? {ctrl}  "
          f"(Kalman rank {rank} / {A.shape[0]})")

    K, X = jaxctrl.lqr(A, B, Q, R)
    Acl = A - B @ K
    print(f"LQR gain K            = {jnp.asarray(K).ravel()}")
    print(f"closed-loop spectrum  : max Re(lambda) = {float(jnp.max(jnp.linalg.eigvals(Acl).real)):.3f}  "
          f"(< 0  -> oscillation quenched)")
    print(f"LQR cost J = trace(X) = {float(jnp.trace(X)):.4f}")

    # (a) linearised closed loop: regulate the deviation y = x - x* to zero
    y0 = jnp.array([5.0, -3.0, 1.0, 0.0, 0.0, 0.0])  # perturbation off x*
    ts, ys, us = jaxctrl.simulate_closed_loop(A, B, K, x0=y0, T=20.0, num_steps=200)
    print(f"linearised closed loop: ||x - x*||  {float(jnp.linalg.norm(ys[0])):.3f}  ->  "
          f"{float(jnp.linalg.norm(ys[-1])):.2e}  over T = 20")

    # (b) the same LQR law u = -K (x - x*) applied to the *nonlinear* repressilator
    try:
        import diffrax as dfx
        x0 = xs + y0
        def closed_loop_nl(t, x, _):
            return repressilator_rhs(x, HILL_N) + (B @ (-K @ (x - xs))).ravel()
        sol = dfx.diffeqsolve(
            dfx.ODETerm(closed_loop_nl), dfx.Tsit5(), t0=0.0, t1=40.0, dt0=0.01,
            y0=x0, saveat=dfx.SaveAt(ts=jnp.linspace(0.0, 40.0, 200)),
            max_steps=20000,
        )
        amp = lambda traj: float(jnp.ptp(traj[-50:, 0]))  # peak-to-peak of m1, late window
        sol_open = dfx.diffeqsolve(
            dfx.ODETerm(lambda t, x, _: repressilator_rhs(x, HILL_N)), dfx.Tsit5(),
            t0=0.0, t1=40.0, dt0=0.01, y0=x0,
            saveat=dfx.SaveAt(ts=jnp.linspace(0.0, 40.0, 200)), max_steps=20000,
        )
        print(f"nonlinear repressilator: m1 oscillation amplitude (late)  "
              f"open loop {amp(sol_open.ys):.2f}  ->  under LQR feedback {amp(sol.ys):.3f}")
    except Exception as e:  # diffrax not installed, or solver hiccup
        print(f"(nonlinear closed-loop sim skipped: {e!r})")

    # parameter sensitivity: d(LQR cost)/d(Hill coefficient n)
    g_ad = float(jax.grad(lqr_cost)(HILL_N))
    g_fd = _finite_diff(lambda n: float(lqr_cost(n)), HILL_N)
    print(f"dJ/dn  autodiff = {g_ad:+.4f}   finite-diff = {g_fd:+.4f}   "
          f"|diff| = {abs(g_ad - g_fd):.2e}")
    print("(positive => stronger cooperativity makes the oscillator costlier to stabilise)")


if __name__ == "__main__":
    main()
