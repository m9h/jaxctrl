# Copyright 2024 jaxctrl contributors. Apache-2.0 license.
"""Lévy area / irreversible circulation — the rough-path twin of the solenoidal drift.

The antisymmetric part of a path's level-2 signature is the **Lévy area**
``A^{ij} = ½(∫X^i dX^j − ∫X^j dX^i)`` — the signed enclosed area / net circulation
of the trajectory in the ``(i,j)`` plane, and the leading time-asymmetric (odd)
coordinate of the log-signature (it flips sign under time reversal).  It is a
model-free, reparametrization-invariant, intrinsically multichannel measure of
broken detailed balance (physics: "areal velocity" / "irreversible circulation").

For a linear Langevin ``dz = Az dt + √(2D) dW`` (stable ``A``; stationary
covariance ``Σ`` from the Lyapunov equation ``AΣ + ΣAᵀ + 2D = 0``) the expected
Lévy-area production *rate* is the Tomita-Tomita irreversible-circulation matrix

    α* = ½(AΣ − ΣAᵀ) = A_sol · Σ ,   A_sol = A + DΣ⁻¹     (Tomita-Tomita 1974; Strang 2024)

and the entropy-production rate ``tr(A_sol Σ A_solᵀ D⁻¹) = tr(α* Σ⁻¹ α*ᵀ D⁻¹)`` is a
weighted ‖α*‖².  Empirically, the time-averaged **Itô** (left-endpoint) Lévy-area
rate of an OU path converges to ``½(ΣAᵀ − AΣ) = −α*`` — so the path estimator
(:func:`levy_rate`) and the Langevin estimator (:func:`solenoidal_circulation`)
are the same matrix up to the Itô sign convention, which :func:`expected_levy_rate`
exposes directly.

References
----------
  Lyons (1998); Friz & Victoir (2010) — rough paths / the area process.
  Chevyrev & Kormilitzin (2016), arXiv:1603.03788 — Lévy area in ML.
  Tomita & Tomita (1974), Prog. Theor. Phys. 51:1731 — irreversible circulation.
  Strang (2024), Axioms 13(12):820 — area-production rate as an OU NESS statistic.
"""

from __future__ import annotations

import jax.numpy as jnp


def levy_area(path: jnp.ndarray) -> jnp.ndarray:
    """Lévy-area matrix of a path — antisymmetric level-2 signature.

    Parameters
    ----------
    path : (T, k) array — multichannel trajectory (time along axis 0).

    Returns
    -------
    A : (k, k) antisymmetric array, ``A^{ij} = ½(∫X^i dX^j − ∫X^j dX^i)`` via the
        discrete left-endpoint (Itô) iterated integral.  Sign convention:
        anticlockwise circulation in the ``(i,j)`` plane is positive (a unit
        circle traversed once gives ``A^{01} = π``).
    """
    path = jnp.asarray(path)
    X = path - path[:1]
    dX = jnp.diff(path, axis=0)
    S = X[:-1].T @ dX                       # S^{ij} = Σ_t X^i_t (X^j_{t+1} − X^j_t)
    return 0.5 * (S - S.T)


def levy_rate(path: jnp.ndarray, dt: float = 1.0) -> jnp.ndarray:
    """Lévy area per unit time (areal velocity) of a path; (k, k) antisymmetric.

    The intensive (rate) form of :func:`levy_area`; for an OU path this converges
    to ``½(ΣAᵀ − AΣ) = −α*`` (see :func:`expected_levy_rate`).
    """
    path = jnp.asarray(path)
    total_time = (path.shape[0] - 1) * dt
    return levy_area(path) / total_time


def circulation_strength(path: jnp.ndarray, dt: float = 1.0) -> jnp.ndarray:
    """Scalar net-circulation strength ‖levy_rate(path)‖_F (Frobenius norm)."""
    return jnp.linalg.norm(levy_rate(path, dt))


def expected_levy_rate(A: jnp.ndarray, Sigma: jnp.ndarray) -> jnp.ndarray:
    """Expected Itô Lévy-area rate of a linear-Langevin path: ``½(ΣAᵀ − AΣ)``.

    This is exactly what :func:`levy_rate` converges to for an OU process with
    drift ``A`` and stationary covariance ``Σ`` — and equals ``−α*`` (the negative
    of :func:`solenoidal_circulation`)."""
    A = jnp.asarray(A)
    Sigma = jnp.asarray(Sigma)
    return 0.5 * (Sigma @ A.T - A @ Sigma)


def solenoidal_circulation(A_sol: jnp.ndarray, Sigma: jnp.ndarray) -> jnp.ndarray:
    """Tomita-Tomita irreversible-circulation matrix ``α* = A_sol · Σ = ½(AΣ − ΣAᵀ)``.

    The model-based expected Lévy-area production rate; pass the solenoidal drift
    ``A_sol`` (``langevin_solenoidal_part``) and stationary covariance ``Σ``.  Equals
    ``−expected_levy_rate(A, Σ)``; detailed balance ⇔ ``α* = 0``."""
    A_sol = jnp.asarray(A_sol)
    Sigma = jnp.asarray(Sigma)
    return A_sol @ Sigma
