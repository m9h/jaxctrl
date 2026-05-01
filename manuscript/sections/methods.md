---
category: research
section: methods
weight: 20
title: "Methods"
status: draft
---

# Methods

jaxctrl is organised in four layers, each filling a gap between SciPy-control
and modern JAX-native autodiff ecosystems.

**Layer 0 — System identification.** `SINDyOptimizer` recovers sparse governing
equations from trajectory data via a sequentially-thresholded least-squares
optimisation; `KoopmanEstimator` implements Exact DMD for linear-in-features
forecasting. Library functions `polynomial_library` and `fourier_library` build
candidate feature dictionaries.

**Layer 1 — Control primitives.** Continuous and discrete Lyapunov solvers use
Bartels–Stewart on a Schur factorisation; the algebraic Riccati equation is
solved via the Hamiltonian Schur method (CARE) and the doubling algorithm
(DARE). Backward passes implement the implicit-derivative identities of
@kao2020autodiff so that `jax.grad` flows through `solve_continuous_are`,
`lqr`, and the Lyapunov solvers without unrolling.

**Layer 2 — Tensor algebra and multilinear control.** Z- and H-eigenvalues are
computed via shifted symmetric higher-order power iteration. The Algebraic
Riccati Tensor Equation (ARTE) is solved by the matricization approach
introduced by @wang2024arte: the system tensor is unfolded along its first
mode, the resulting matrix CARE is solved, and the solution is folded back.
An optional Newton refinement is available when Optimistix is installed.

**Layer 3 — Hypergraph control.** Adjacency and Laplacian tensors are built
from an `hgx` hypergraph; the tensor Kalman rank test of @chen2021hypergraph
generalises classical controllability to higher-order interactions, with
extensions to temporal hypergraphs following @dong2024temporal.

All solvers are JIT-compiled and respect float64 when enabled, and are tested
without `unittest.mock` to keep numerical paths under direct comparison.
