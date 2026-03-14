"""Shared test fixtures for jaxctrl tests."""

import jax
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
