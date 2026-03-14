# jaxctrl

Differentiable control theory in JAX. Lyapunov and Riccati solvers, controllability analysis, tensor eigenvalues, and hypergraph control — all JIT-compiled and autodiff-compatible.

Built on the [Kidger stack](https://docs.kidger.site): Equinox, Lineax, Optimistix, Diffrax.

## Installation

```bash
pip install jaxctrl
```

For hypergraph control (requires [hgx](https://github.com/m9h/hgx)):
```bash
pip install jaxctrl[hypergraph]
```

## Architecture

**Layer 1 — Control primitives** (missing from JAX, exist in SciPy):
- `solve_continuous_lyapunov`, `solve_discrete_lyapunov`
- `solve_continuous_are`, `solve_discrete_are`
- `lqr`, `dlqr`
- `controllability_gramian`, `observability_gramian`
- `is_controllable`, `is_observable`, `is_stabilizable`, `is_detectable`

**Layer 2 — Tensor control** (new mathematics, no implementation exists anywhere):
- `z_eigenvalues`, `h_eigenvalues`, `spectral_radius`
- `tensor_unfold`, `tensor_fold`, `einstein_product`, `tensor_contract`
- `solve_arte`, `tensor_lyapunov`, `multilinear_lqr`

**Layer 3 — Hypergraph control** (integrates with hgx):
- `adjacency_tensor`, `laplacian_tensor`
- `tensor_kalman_rank`, `minimum_driver_nodes`
- `control_energy`, `controllability_profile`
- `HypergraphControlSystem`

## Quick start

```python
import jax.numpy as jnp
import jaxctrl

# Double integrator: dx/dt = Ax + Bu
A = jnp.array([[0.0, 1.0], [0.0, 0.0]])
B = jnp.array([[0.0], [1.0]])
Q = jnp.eye(2)
R = jnp.eye(1)

# LQR controller (fully differentiable)
K, X = jaxctrl.lqr(A, B, Q, R)

# Controllability analysis
print(jaxctrl.is_controllable(A, B))  # True

# Hypergraph control (requires hgx)
import hgx
hg = hgx.from_incidence(jnp.ones((5, 1)))
A_sys, B_sys = jaxctrl.hypergraph_linear_system(hg, driver_nodes=jnp.array([0]))
```

## References

- Kao & Hennequin (2020). "Automatic differentiation of Sylvester, Lyapunov, and algebraic Riccati equations." [arXiv:2011.11430](https://arxiv.org/abs/2011.11430)
- Chen & Surana (2021). "Controllability of hypergraphs." IEEE TNSE.
- Wang & Wei (2024). "Algebraic Riccati tensor equations." [arXiv:2402.13491](https://arxiv.org/abs/2402.13491)
- Dong et al. (2024). "Controllability and observability of temporal hypergraphs." [arXiv:2408.12085](https://arxiv.org/abs/2408.12085)
- Liu, Slotine & Barabási (2011). "Controllability of complex networks." Nature 473, 167–173.
