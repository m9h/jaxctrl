"""Shared test fixtures for jaxctrl tests."""

import jax

# Enable float64 globally for tests — eigendecomposition-based solvers
# (CARE, DARE) require double precision for reliable FD gradient checks.
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import pytest


@pytest.fixture
def key():
    """A fixed PRNG key for reproducible tests."""
    return jax.random.PRNGKey(42)


@pytest.fixture
def double_integrator():
    """The double-integrator system: x'' = u.

    State-space form:
        A = [[0, 1], [0, 0]],  B = [[0], [1]]

    This is the canonical test system in control theory.
    """
    A = jnp.array([[0.0, 1.0], [0.0, 0.0]])
    B = jnp.array([[0.0], [1.0]])
    return A, B
