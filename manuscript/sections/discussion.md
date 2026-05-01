---
category: research
section: discussion
weight: 40
title: "Discussion"
status: draft
---

# Discussion

jaxctrl fills a specific gap: SciPy provides classical control primitives but
no autodiff path; the JAX ecosystem provides autodiff but no control library.
Diffrax, Lineax, and Optimistix supply ODE integration, linear solves, and
nonlinear roots respectively, yet there is no JAX-native equivalent of
`scipy.signal` or `python-control`. By implementing CARE, DARE, and
Lyapunov solvers with custom VJPs that follow the implicit-derivative
identities of @kao2020autodiff, we make the entire LQR pipeline composable
with the rest of the JAX stack — including `jax.vmap`, `jax.jit`, and
neural-network parameterisations of the cost matrices.

The tensor-control layer takes this further. The classical controllability
result of @liu2011controllability — that the structure of a directed network
determines the minimum number of driver nodes — assumes pairwise
interactions. Many real networks (metabolic, neural, social) involve
genuinely higher-order interactions for which a hypergraph representation
is more faithful. @chen2021hypergraph established a tensor Kalman rank test
that extends controllability to this setting, and @dong2024temporal extended
it to time-varying hypergraphs. jaxctrl's Layer 3 implements both, with the
ARTE solver of @wang2024arte providing a multilinear LQR primitive on top.

Limitations. The matricization approach to ARTE is exact only for tensor
systems that factor through their mode-1 unfolding; the general case
remains an open problem and is treated as a first-order approximation
with optional Newton refinement. Hypergraph controllability tests scale
combinatorially in the number of driver-node candidates and are intended
for networks of order $10^2$ to $10^4$ rather than internet-scale graphs.

Future work. The autoresearch loop will systematically explore where the
differentiable formulation pays off — most likely in cost-matrix tuning,
end-to-end learning of system identification + control, and gradient-based
search over driver-node placements.
