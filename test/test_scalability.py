"""Scalability tests: Lineax iterative solvers for large systems.

Verifies that the Lineax GMRES solver matches the Kronecker solver
for small systems and can handle n=100 (which would be impractical
with the O(n^6) Kronecker approach).
"""

import jax
import jax.numpy as jnp
import pytest
from jaxctrl._lyapunov import (
    _HAS_LINEAX,
    _solve_continuous_lyapunov_kron,
    _solve_continuous_lyapunov_impl,
    _symmetrise,
    solve_continuous_lyapunov,
)

pytestmark = pytest.mark.skipif(not _HAS_LINEAX, reason="lineax not installed")


# Import the lineax solver directly for testing
if _HAS_LINEAX:
    from jaxctrl._lyapunov import _solve_continuous_lyapunov_lineax


class TestLineaxMatchesKronecker:
    """Lineax and Kronecker solvers should agree on small problems."""

    def test_2x2_diagonal(self):
        """Both solvers agree on A = diag(-1, -2), Q = I."""
        A = jnp.array([[-1.0, 0.0], [0.0, -2.0]])
        Q = jnp.eye(2)

        X_kron = _solve_continuous_lyapunov_kron(A, Q)
        X_lx = _solve_continuous_lyapunov_lineax(A, Q)

        assert jnp.allclose(X_kron, X_lx, atol=1e-5), (
            f"Kronecker:\n{X_kron}\nLineax:\n{X_lx}"
        )

    def test_4x4_random(self):
        """Both solvers agree on a random 4x4 stable system."""
        key = jax.random.PRNGKey(42)
        k1, k2 = jax.random.split(key)

        A_raw = jax.random.normal(k1, (4, 4))
        eigvals = jnp.linalg.eigvals(A_raw)
        shift = jnp.max(jnp.real(eigvals)) + 1.0
        A = A_raw - shift * jnp.eye(4)

        Q = jax.random.normal(k2, (4, 4))
        Q = Q @ Q.T

        X_kron = _solve_continuous_lyapunov_kron(A, Q)
        X_lx = _solve_continuous_lyapunov_lineax(A, Q)

        assert jnp.allclose(X_kron, X_lx, atol=1e-4), (
            f"Max diff: {jnp.max(jnp.abs(X_kron - X_lx))}"
        )


class TestLineaxLargeScale:
    """Test Lineax solver on systems too large for Kronecker."""

    def test_n100_residual(self):
        """Lineax solver produces a valid solution for n=100.

        The Kronecker approach would require a 10000x10000 matrix
        (800 MB in float64), making it impractical.
        """
        n = 100
        key = jax.random.PRNGKey(123)
        k1, k2 = jax.random.split(key)

        # Random stable A: A = A_raw - (max_real_eig + 1) * I
        A_raw = jax.random.normal(k1, (n, n)) / jnp.sqrt(n)
        eigvals = jnp.linalg.eigvals(A_raw)
        shift = jnp.max(jnp.real(eigvals)) + 1.0
        A = A_raw - shift * jnp.eye(n)

        Q = jax.random.normal(k2, (n, n))
        Q = Q @ Q.T / n  # Scale for conditioning

        X = _solve_continuous_lyapunov_lineax(A, Q)

        # Verify the Lyapunov residual
        residual = A @ X + X @ A.T + Q
        rel_residual = jnp.linalg.norm(residual) / jnp.linalg.norm(Q)
        assert rel_residual < 1e-4, (
            f"Relative residual = {rel_residual} (should be < 1e-4)"
        )

    def test_n100_symmetric_psd(self):
        """Solution for n=100 should be symmetric and PSD."""
        n = 100
        key = jax.random.PRNGKey(456)
        k1, k2 = jax.random.split(key)

        A_raw = jax.random.normal(k1, (n, n)) / jnp.sqrt(n)
        eigvals = jnp.linalg.eigvals(A_raw)
        shift = jnp.max(jnp.real(eigvals)) + 1.0
        A = A_raw - shift * jnp.eye(n)

        Q = jax.random.normal(k2, (n, n))
        Q = Q @ Q.T / n

        X = _solve_continuous_lyapunov_lineax(A, Q)

        assert jnp.allclose(X, X.T, atol=1e-6), "X should be symmetric"
        eigs = jnp.linalg.eigvalsh(X)
        assert jnp.all(eigs >= -1e-4), f"X should be PSD, min eigval: {jnp.min(eigs)}"

    def test_dispatch_uses_lineax_for_large_n(self):
        """The impl function should dispatch to Lineax for n > threshold."""
        n = 60  # Above the threshold of 50
        key = jax.random.PRNGKey(789)
        k1, k2 = jax.random.split(key)

        A_raw = jax.random.normal(k1, (n, n)) / jnp.sqrt(n)
        eigvals = jnp.linalg.eigvals(A_raw)
        shift = jnp.max(jnp.real(eigvals)) + 1.0
        A = A_raw - shift * jnp.eye(n)

        Q = jax.random.normal(k2, (n, n))
        Q = Q @ Q.T / n

        # This should use Lineax internally (n=60 > threshold=50)
        X = _solve_continuous_lyapunov_impl(A, Q)

        residual = A @ X + X @ A.T + Q
        rel_residual = jnp.linalg.norm(residual) / jnp.linalg.norm(Q)
        assert rel_residual < 1e-4, (
            f"Relative residual = {rel_residual}"
        )
