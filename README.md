---
category: research
section: introduction
weight: 10
title: "jaxctrl: Differentiable Control Theory in JAX"
status: draft
slide_summary: "Fully differentiable Lyapunov/Riccati solvers, tensor eigenvalue methods, and hypergraph controllability analysis in JAX — filling gaps between SciPy control and modern autodiff ecosystems."
tags: [jax, control-theory, lyapunov, riccati, tensor-control, hypergraph, differentiable, system-identification]
---

# jaxctrl

Differentiable control theory in JAX. Lyapunov and Riccati solvers, controllability analysis, tensor eigenvalues, and hypergraph control — all JIT-compiled and autodiff-compatible.

Built on the [Kidger stack](https://docs.kidger.site): Equinox, Lineax, Optimistix, Diffrax.

## Installation

```bash
pip install jaxctrl
```

Optional extras:

- `pip install jaxctrl[solvers]` — pulls in [Lineax](https://github.com/patrick-kidger/lineax) and [Optimistix](https://github.com/patrick-kidger/optimistix). Enables the iterative Lyapunov solver for large systems (n > 50) and Newton refinement for the ARTE solver.
- `pip install jaxctrl[diffrax]` — pulls in [Diffrax](https://github.com/patrick-kidger/diffrax). Enables adaptive ODE integration in `simulate_lti` and `simulate_closed_loop` (a matrix-exponential fallback is used otherwise).
- `pip install jaxctrl[hypergraph]` — pulls in [hgx](https://github.com/m9h/hgx). Enables the Layer 3 hypergraph controllability stack.

## Architecture

**Layer 0 — System identification** (data-driven model discovery):
- `SINDyOptimizer`, `polynomial_library`, `fourier_library`
- `KoopmanEstimator` (Exact DMD)

**Layer 1 — Control primitives** (missing from JAX, exist in SciPy):
- `solve_continuous_lyapunov`, `solve_discrete_lyapunov`
- `solve_continuous_are`, `solve_discrete_are`
- `lqr`, `dlqr`
- `controllability_gramian`, `observability_gramian`
- `is_controllable`, `is_observable`, `is_stabilizable`, `is_detectable`
- `simulate_lti`, `simulate_closed_loop` (Diffrax adaptive ODE or matrix-exponential fallback)

**Layer 2 — Tensor control** (new mathematics, no implementation exists anywhere):
- `z_eigenvalues`, `h_eigenvalues`, `spectral_radius`
- `tensor_unfold`, `tensor_fold`, `einstein_product`, `tensor_contract`
- `mode_dot`, `hosvd`, `tucker_to_tensor`, `khatri_rao`
- `solve_arte`, `tensor_lyapunov`, `multilinear_lqr`

**Layer 3 — Hypergraph control** (integrates with hgx):
- `adjacency_tensor`, `laplacian_tensor`
- `tensor_kalman_rank`, `minimum_driver_nodes`
- `control_energy`, `controllability_profile`
- `HypergraphControlSystem`

## Quick start

```python
import jax
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

# Simulate closed-loop response (uses Diffrax if available)
x0 = jnp.array([2.0, 0.0])
ts, xs, us = jaxctrl.simulate_closed_loop(A, B, K, x0, T=10.0)

# Differentiate the LQR cost w.r.t. Q
dJ_dQ = jax.grad(lambda Q: jnp.sum(jaxctrl.lqr(A, B, Q, R)[1]))(Q)
```

## Applications: gene-regulatory networks & cellular dynamics

GRNs and cellular dynamics map cleanly onto the four layers — the whole *"identify a surrogate
model → do control theory on it"* pipeline is what Layers 0–1 are for, and the hypergraph layer
(Layer 3) is built directly on the Liu–Slotine–Barabási / Chen–Surana network-controllability
line that originated in systems biology.

| Layer | jaxctrl | Cellular-systems use | Example task |
|---|---|---|---|
| **L0** | `SINDyOptimizer`, `polynomial_library`, `KoopmanEstimator` (DMD) | Discover an ODE / Koopman model from gene-expression or signaling time series | Recover regulatory ODEs from perturbation time courses; DMD on RNA-velocity vector fields |
| **L1** | `lqr`/`dlqr`, `is_controllable`/`is_stabilizable`, `*_gramian`, `simulate_closed_loop` | Linearise around a fixed point / limit cycle; ask which genes are steerable, design a "drug input" | Steer the cell cycle / p53 / NF-κB to a target state; controllability of a linearised GRN |
| **L2** | `solve_arte`, `tensor_lyapunov`, `multilinear_lqr`, `z_`/`h_eigenvalues` | Higher-order regulation (TF-complex / cooperative binding → 3-way terms), bilinear control | Multilinear LQR on a GRN with quadratic Hill-type couplings |
| **L3** | `adjacency_tensor`, `minimum_driver_nodes`, `control_energy`, `HypergraphControlSystem` | GRN as a hypergraph: a TF complex regulating a gene module = one hyperedge → minimum driver-gene set, control-energy landscape | "Which TFs must I perturb to control this regulon?" on RegulonDB / YEASTRACT topology |

**Datasets & benchmarks that fit** (smallest-first):

- *Tiny synthetic GRNs (known ground truth, n ≤ ~10)* — **repressilator** (Elowitz & Leibler 2000;
  3-gene ring oscillator — see [`examples/repressilator_control_demo.py`](examples/repressilator_control_demo.py)),
  **toggle switch** (Gardner et al. 2000; 2-gene bistable — drive between attractors),
  **IRMA** (Cantone et al. 2009; 5-gene yeast inference benchmark with galactose on/off time series — ideal `SINDyOptimizer` → `lqr` demo),
  **E. coli SOS network** (~8 genes; Uri Alon lab).
- *In-silico suites with ground-truth topology* — **DREAM4/5** (GeneNetWeaver, size-10/100 networks + time-series/knockout data), **SERGIO** (Dibaeinia & Sinha 2020), **BoolODE / BEELINE** (Pratapa et al. 2020) — L0 to recover dynamics, L3 to compute `minimum_driver_nodes` vs the true topology.
- *Real network topologies (L3 driver-node side)* — **RegulonDB** (E. coli TF→gene), **YEASTRACT** (S. cerevisiae) — sigma factors / TF complexes become hyperedges → `minimum_driver_nodes` / `controllability_profile`; the **yeast cell-cycle network** (Li et al. 2004 / Davidich & Bornholdt).
- *Single-cell / continuous trajectories (L0 Koopman/DMD)* — **dynamo** (Qiu et al. 2022), RNA-velocity / **CellRank** datasets — fit a linear operator on the same trajectories, then L1 controllability on it.
- *Well-characterised ODE models (skip L0 → L1/L2)* — MAPK/ERK, p53–Mdm2, NF-κB, circadian (Goldbeter), cell-cycle (Tyson–Novák) — published SBML in BioModels; linearise → `lqr` + `controllability_gramian` + `jax.grad` for parameter-sensitivity of controllability.

**Caveat on fit.** jaxctrl is *linear / multilinear* control — not full nonlinear MPC, the chemical
master equation, or Boolean-network dynamics natively. The realistic workflow is always:
*(L0 or hand-derived) linear / Koopman / multilinear surrogate → (L1–L3) controllability + LQR +
driver nodes → `jax.grad` for sensitivities*. For Boolean GRNs, take a continuous relaxation first.
(Downstream, e.g. in [`anatomical-compiler`](https://github.com/m9h/anatomical-compiler), jaxctrl is
the controller-synthesis layer on top of a learned Hypergraph Neural ODE surrogate.)

## References

- Kao & Hennequin (2020). "Automatic differentiation of Sylvester, Lyapunov, and algebraic Riccati equations." [arXiv:2011.11430](https://arxiv.org/abs/2011.11430)
- Elowitz & Leibler (2000). "A synthetic oscillatory network of transcriptional regulators." Nature 403, 335–338.
- Chen & Surana (2021). "Controllability of hypergraphs." IEEE TNSE.
- Wang & Wei (2024). "Algebraic Riccati tensor equations." [arXiv:2402.13491](https://arxiv.org/abs/2402.13491)
- Dong et al. (2024). "Controllability and observability of temporal hypergraphs." [arXiv:2408.12085](https://arxiv.org/abs/2408.12085)
- Liu, Slotine & Barabási (2011). "Controllability of complex networks." Nature 473, 167–173.
