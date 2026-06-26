# Copyright 2024 jaxctrl contributors. Apache-2.0 license.
"""Langevin / Fokker-Planck system identification.

Recovers the DRIFT and DIFFUSION of a stochastic process from a trajectory, and
splits the drift into its dissipative (gradient) and solenoidal (rotational,
irreversible) parts with the entropy-production rate — the data-driven member of
the Fokker-Planck family (Friston's f = (Γ+Q)∇log p; Ingber's SMNI).

This captures what DMD/SINDy/DYSCO discard: the diffusion D(z) and the solenoidal
flow that breaks detailed balance. A non-equilibrium cycle shows up here as a
nonzero solenoidal part + positive entropy production, even when the deterministic
drift has zero net rotation — see docs/LANGEVIN_FOKKER_PLANCK_NEUROIMAGING.md.

This first estimator is the linear / Gaussian (Ornstein-Uhlenbeck) case, for which
the Helmholtz split and entropy production are closed-form:

    dz = A z dt + sqrt(2D) dW,   stationary covariance Σ
    gradient (reversible) drift   A_rev = -D Σ⁻¹      (detailed balance)
    solenoidal (irreversible)     A_sol = A - A_rev = A + D Σ⁻¹
    entropy production rate        Ṡ = tr(A_sol Σ A_solᵀ D⁻¹) ≥ 0,  = 0 iff A_sol = 0
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


class LinearLangevin(NamedTuple):
    """Fitted linear Langevin model ``dz = A z dt + sqrt(2D) dW``."""

    A: jax.Array       # drift matrix (k, k)
    D: jax.Array       # diffusion matrix (k, k)
    Sigma: jax.Array   # stationary covariance (k, k)
    dt: float


def fit_linear_langevin(Z: jax.Array, dt: float) -> LinearLangevin:
    """Estimate drift A and diffusion D from a trajectory ``Z`` of shape (T, k).

    Drift via the first Kramers-Moyal moment (least-squares ż = A z); diffusion
    via the second moment of the increments / (2 dt); Σ from the empirical
    covariance.
    """
    Z = jnp.asarray(Z)
    Z = Z - jnp.mean(Z, axis=0)
    inc = Z[1:] - Z[:-1]
    Zc = Z[:-1]
    dZ = inc / dt

    # drift: dZ ≈ Zc @ A.T  ->  A = ((Zc.T Zc)^-1 Zc.T dZ).T
    G = Zc.T @ Zc
    A = jnp.linalg.solve(G, Zc.T @ dZ).T

    # diffusion: 2nd KM moment of the increments
    D = (inc.T @ inc) / inc.shape[0] / (2.0 * dt)
    Sigma = (Z.T @ Z) / Z.shape[0]
    return LinearLangevin(A=A, D=D, Sigma=Sigma, dt=dt)


def langevin_gradient_part(m: LinearLangevin) -> jax.Array:
    """Dissipative / reversible (detailed-balance) drift A_rev = -D Σ⁻¹."""
    return -m.D @ jnp.linalg.inv(m.Sigma)


def langevin_solenoidal_part(m: LinearLangevin) -> jax.Array:
    """Solenoidal / irreversible (cyclic) drift A_sol = A - A_rev. Nonzero ⇔
    broken detailed balance ⇔ a non-equilibrium probability current."""
    return m.A - langevin_gradient_part(m)


def langevin_entropy_production(m: LinearLangevin) -> jax.Array:
    """Entropy-production rate Ṡ = tr(A_sol Σ A_solᵀ D⁻¹) ≥ 0 (0 ⇔ equilibrium)."""
    A_sol = langevin_solenoidal_part(m)
    return jnp.trace(A_sol @ m.Sigma @ A_sol.T @ jnp.linalg.inv(m.D))


def langevin_solenoidal_frequency(m: LinearLangevin) -> jax.Array:
    """Rotation frequency (Hz) of the solenoidal flow = |Im λ(A_sol)|/2π.

    This is the cyclic frequency carried by the *probability current*; it can be
    nonzero even when the deterministic drift A has no oscillatory eigenvalues —
    i.e. the stochastic cycle that DMD/SINDy/DYSCO miss.
    """
    ev = jnp.linalg.eigvals(langevin_solenoidal_part(m))
    return jnp.max(jnp.abs(jnp.imag(ev))) / (2.0 * jnp.pi)


# ---------------------------------------------------------------------------
# Discrete-state (jump-process) version — for HMM/DyNeMo state sequences
# ---------------------------------------------------------------------------
#
# The continuous estimator needs a continuous trajectory whose diffusion is in
# band.  For a coarse-grained discrete state sequence (the natural substrate for
# brain-network HMM states), the non-equilibrium structure is the asymmetry of
# the transition statistics (Schnakenberg / Lynn et al. 2021): the net flux
# F_ij = N_ij - N_ji and the entropy production Σ P_ij log(P_ij / P_ji).


def transition_flux(states: jax.Array, n_states: int) -> jax.Array:
    """Net transition-flux matrix F_ij = N_ij - N_ji (antisymmetric).  A nonzero
    flux that circulates among states is the discrete probability current."""
    s = jnp.asarray(states)
    idx = s[:-1] * n_states + s[1:]
    N = jnp.bincount(idx, length=n_states * n_states).reshape(n_states, n_states)
    return N - N.T


def discrete_entropy_production(states: jax.Array, n_states: int) -> jax.Array:
    """Entropy-production rate of a discrete state sequence,
    Ṡ = Σ_ij P_ij log(P_ij / P_ji) ≥ 0 (0 ⇔ detailed balance), over transition
    pairs observed in both directions.  P_ij = joint prob of a consecutive i→j."""
    s = jnp.asarray(states)
    idx = s[:-1] * n_states + s[1:]
    N = jnp.bincount(idx, length=n_states * n_states).reshape(n_states, n_states)
    P = N / jnp.sum(N)
    Pt = P.T
    mask = (P > 0) & (Pt > 0)
    return jnp.sum(jnp.where(mask, P * jnp.log(jnp.where(mask, P / jnp.where(Pt > 0, Pt, 1.0), 1.0)), 0.0))
