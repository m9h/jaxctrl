---
category: research
section: introduction
weight: 10
title: "jaxctrl: Differentiable Control Theory in JAX"
status: draft
slide_summary: "Fully differentiable Lyapunov/Riccati solvers, tensor eigenvalue methods, and hypergraph controllability analysis in JAX — filling gaps between SciPy control and modern autodiff ecosystems."
tags: [jax, control-theory, lyapunov, riccati, tensor-control, hypergraph, differentiable, system-identification]
---

<div align="center">

# 🎛️ jaxctrl

### Differentiable control theory in JAX

**Lyapunov & Riccati solvers · controllability analysis · tensor eigenvalues · hypergraph control** — all `jit`-compiled, `vmap`-able, and end-to-end autodiff-friendly.

[![CI](https://github.com/m9h/jaxctrl/actions/workflows/ci.yml/badge.svg)](https://github.com/m9h/jaxctrl/actions/workflows/ci.yml)
&nbsp;![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)
&nbsp;![Built on JAX](https://img.shields.io/badge/built%20on-JAX-orange.svg)
&nbsp;![License](https://img.shields.io/badge/license-Apache--2.0-brightgreen.svg)
&nbsp;[![Code style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

*Built on the [Kidger stack](https://docs.kidger.site) — Equinox · Lineax · Optimistix · Diffrax.*

</div>

---

> **Why jaxctrl?** SciPy has the classical control solvers but no autodiff; JAX has autodiff but no control solvers. jaxctrl closes the gap — and then pushes past it into *tensor* and *hypergraph* control, where (as far as we know) no other implementation exists.

## 📦 Installation

```bash
pip install jaxctrl
```

Optional extras:

| Extra | `pip install jaxctrl[...]` | Pulls in | Enables |
|---|---|---|---|
| 🧰 `solvers` | `jaxctrl[solvers]` | [Lineax](https://github.com/patrick-kidger/lineax), [Optimistix](https://github.com/patrick-kidger/optimistix) | Iterative Lyapunov solver for large systems (`n > 50`) and Newton refinement for the ARTE solver |
| 🌊 `diffrax` | `jaxctrl[diffrax]` | [Diffrax](https://github.com/patrick-kidger/diffrax) | Adaptive ODE integration in `simulate_lti` / `simulate_closed_loop` (matrix-exponential fallback otherwise) |
| 🕸️ `hypergraph` | `jaxctrl[hypergraph]` | [hgx](https://github.com/m9h/hgx) | The Layer 3 hypergraph controllability stack |

## 🏗️ Architecture

A four-layer stack — each layer builds on the one below, and every primitive is JIT-compilable and differentiable.

### 🧪 Layer 0 — System identification  ·  *data-driven model discovery*
- `SINDyOptimizer`, `polynomial_library`, `fourier_library`
- `KoopmanEstimator` (Exact DMD)

### 🎚️ Layer 1 — Control primitives  ·  *the SciPy-control gap, now differentiable*
- `solve_continuous_lyapunov`, `solve_discrete_lyapunov`
- `solve_continuous_are`, `solve_discrete_are`
- `lqr`, `dlqr`
- `controllability_gramian`, `observability_gramian`
- `is_controllable`, `is_observable`, `is_stabilizable`, `is_detectable`
- `simulate_lti`, `simulate_closed_loop` (Diffrax adaptive ODE or matrix-exponential fallback)

### 🧮 Layer 2 — Tensor control  ·  *new mathematics — no other implementation exists*
- `z_eigenvalues`, `h_eigenvalues`, `spectral_radius`
- `tensor_unfold`, `tensor_fold`, `einstein_product`, `tensor_contract`
- `mode_dot`, `hosvd`, `tucker_to_tensor`, `khatri_rao`
- `solve_arte`, `tensor_lyapunov`, `multilinear_lqr`

### 🕸️ Layer 3 — Hypergraph control  ·  *higher-order networks (integrates with [hgx](https://github.com/m9h/hgx))*
- `adjacency_tensor`, `laplacian_tensor`
- `tensor_kalman_rank`, `minimum_driver_nodes`
- `control_energy`, `controllability_profile`
- `HypergraphControlSystem`

## 🚀 Quick start

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

## 🗺️ Examples

| File | What it shows |
|---|---|
| [`examples/diff_lqr_demo.py`](examples/diff_lqr_demo.py) | `jax.grad` of the LQR cost w.r.t. the state weight `Q`, cross-checked against finite differences |
| [`examples/tensor_lqr_demo.py`](examples/tensor_lqr_demo.py) | Multilinear LQR via the matricized ARTE solver on an order-3 system tensor |
| [`examples/repressilator_control_demo.py`](examples/repressilator_control_demo.py) | Quenching the repressilator: linearize a 3-gene ring oscillator → controllability → LQR → quench the *nonlinear* oscillation → `jax.grad` w.r.t. the Hill coefficient |
| [`examples/sindy_lqr_demo.ipynb`](examples/sindy_lqr_demo.ipynb) | SINDy model discovery from trajectory data, then LQR on the recovered system |
| [`examples/irma_sindy_lqr.ipynb`](examples/irma_sindy_lqr.ipynb) | A gene-regulatory network end-to-end: simulate an IRMA-topology Hill-ODE → `SINDyOptimizer` linear surrogate → controllability → LQR "drug input" steering the network back to switch-off → `jax.grad` of the control cost w.r.t. a feedback edge |
| [`examples/grn_hypergraph_drivers.ipynb`](examples/grn_hypergraph_drivers.ipynb) | Layer 3 on a GRN-as-hypergraph: `minimum_driver_nodes`, per-TF `controllability_profile`, the `control_energy` landscape over driver sets, and `HypergraphControlSystem` + LQR — "which TFs must I perturb to control this regulon?" |

## 🧬 Applications: gene-regulatory networks & cellular dynamics

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

## 📚 References

- Kao & Hennequin (2020). "Automatic differentiation of Sylvester, Lyapunov, and algebraic Riccati equations." [arXiv:2011.11430](https://arxiv.org/abs/2011.11430)
- Elowitz & Leibler (2000). "A synthetic oscillatory network of transcriptional regulators." Nature 403, 335–338.
- Chen & Surana (2021). "Controllability of hypergraphs." IEEE TNSE.
- Wang & Wei (2024). "Algebraic Riccati tensor equations." [arXiv:2402.13491](https://arxiv.org/abs/2402.13491)
- Dong et al. (2024). "Controllability and observability of temporal hypergraphs." [arXiv:2408.12085](https://arxiv.org/abs/2408.12085)
- Liu, Slotine & Barabási (2011). "Controllability of complex networks." Nature 473, 167–173.
