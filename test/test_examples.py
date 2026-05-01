"""Tests that the example scripts run end-to-end and expose stable entry points."""

from __future__ import annotations

import jax.numpy as jnp


def test_diff_lqr_demo_runs() -> None:
    from examples import diff_lqr_demo

    diff_lqr_demo.main()


def test_tensor_lqr_demo_runs() -> None:
    from examples import tensor_lqr_demo

    tensor_lqr_demo.main()


def test_diff_lqr_gradient_finite() -> None:
    from examples import diff_lqr_demo

    grad = diff_lqr_demo.compute_grad()
    assert bool(jnp.all(jnp.isfinite(grad))), f"non-finite gradient: {grad}"
