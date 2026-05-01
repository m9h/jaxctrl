"""Autoresearch experiment for jaxctrl.

This file is modified by the autonomous research agent.
The agent changes parameters, models, and configurations here.
Infrastructure (imports, metric computation, output format) is fixed.

Baseline experiment
-------------------
Compare the differentiable jaxctrl CARE solver against scipy.linalg's
classical solver on a linearised Van der Pol system around the origin.

The metric defined in program.md is::

    control_advantage = (cost_classical - cost_jaxctrl) / cost_classical

evaluated at matched wall time.  The cost is the LQR value function
J = trace(P @ Sigma0) with Sigma0 = I.

For a baseline run both solvers produce essentially identical P, so the
metric is ~0 — that is the expected starting point.  The autoresearch
agent will perturb the system / cost / solver to find regimes where the
differentiable path beats the classical path.
"""

import json
import sys
import time

# === EXPERIMENT CONFIGURATION (agent modifies this section) ===

EXPERIMENT = {
    "description": "linearised Van der Pol baseline (n=2, mu=1.0)",
    "parameters": {
        "system": "vanderpol_linearised",
        "n": 2,
        "mu": 1.0,
    },
}

# === INFRASTRUCTURE (do not modify below this line) ===


def run_experiment(config: dict) -> dict:
    """Run the LQR-cost comparison and return a result dict."""
    import jax

    jax.config.update("jax_enable_x64", True)

    import jax.numpy as jnp
    import numpy as np
    import scipy.linalg as sla

    import jaxctrl

    params = config["parameters"]
    n = int(params["n"])
    mu = float(params["mu"])

    # Linearised Van der Pol around the origin:
    #   d^2 x/dt^2 - mu (1 - x^2) dx/dt + x = 0
    # Linearised: A = [[0, 1], [-1, mu]] (Jacobian at origin).
    A = jnp.array([[0.0, 1.0], [-1.0, mu]])
    B = jnp.array([[0.0], [1.0]])
    Q = jnp.eye(n)
    R = jnp.eye(1)

    # --- jaxctrl path (jit-compiled, fully differentiable) ---
    care_jit = jax.jit(jaxctrl.solve_continuous_are)
    # Warm up JIT so the timed call measures execution, not compilation.
    P_jax = care_jit(A, B, Q, R)
    P_jax.block_until_ready()

    t0 = time.perf_counter()
    P_jax = care_jit(A, B, Q, R)
    P_jax.block_until_ready()
    wall_time_jaxctrl = time.perf_counter() - t0

    # --- classical path (scipy / LAPACK) ---
    A_np = np.asarray(A)
    B_np = np.asarray(B)
    Q_np = np.asarray(Q)
    R_np = np.asarray(R)

    t0 = time.perf_counter()
    P_scipy = sla.solve_continuous_are(A_np, B_np, Q_np, R_np)
    wall_time_scipy = time.perf_counter() - t0

    Sigma0 = np.eye(n)
    cost_jaxctrl = float(np.trace(np.asarray(P_jax) @ Sigma0))
    cost_scipy = float(np.trace(P_scipy @ Sigma0))

    metric = (cost_scipy - cost_jaxctrl) / cost_scipy

    return {
        "metric_value": metric,
        "parameters": {
            **params,
            "wall_time_jaxctrl": wall_time_jaxctrl,
            "wall_time_scipy": wall_time_scipy,
            "cost_jaxctrl": cost_jaxctrl,
            "cost_scipy": cost_scipy,
        },
        "status": "ok",
    }


def main() -> int:
    try:
        result = run_experiment(EXPERIMENT)
    except Exception as exc:  # noqa: BLE001 — surface as a structured error
        result = {
            "metric_value": float("nan"),
            "parameters": {
                "wall_time_jaxctrl": -1.0,
                "wall_time_scipy": -1.0,
                "error": repr(exc),
            },
            "status": "error",
        }

    print(
        f"RESULT|{result['metric_value']}|"
        f"{json.dumps(result['parameters'])}|"
        f"{result['status']}|"
        f"{EXPERIMENT['description']}"
    )
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
