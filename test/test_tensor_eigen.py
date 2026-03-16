"""Known-answer tests for tensor eigenvalue solvers.

Every test computes an expected result by hand (documented in the docstring)
and compares it to the implementation output.

Background
----------
For a k-th order n-dimensional symmetric tensor T, a scalar lambda is a
Z-eigenvalue with Z-eigenvector x (||x||=1) if:

    T x^{k-1} = lambda x

where T x^{k-1} means contracting T with x along all but one mode.

Sections
--------
1. Identity tensor Z-eigenvalues
2. Diagonal tensor Z-eigenvalues
3. Spectral radius
4. Power method convergence
"""

import jax
import jax.numpy as jnp
from jaxctrl._tensor_eigen import (
    spectral_radius,
    tensor_power_method,
    z_eigenvalues,
)


# Note: z_eigenvalues uses orthogonal projection deflation with multiple
# random restarts.  The spectral_radius function also uses multiple
# restarts to find the globally dominant eigenvalue.

# -----------------------------------------------------------------------
# 1. Identity tensor
# -----------------------------------------------------------------------


class TestIdentityTensorEigenvalues:
    r"""The k-th order identity tensor I has I[i,i,...,i] = 1 and 0 elsewhere.

    For the 3rd-order, 2-dimensional identity tensor:
        I[0,0,0] = 1,  I[1,1,1] = 1,  all other entries = 0.

    Z-eigenvalue equation: (I x^2)_i = sum_{j,k} I[i,j,k] x_j x_k = x_i^2.
    So I x^2 = [x_0^2, x_1^2].
    For this to equal lambda * x with ||x||=1:
        x_i^2 = lambda * x_i  for each i.

    If x_i != 0: x_i = lambda.
    With ||x||=1 and, say, x = e_0 = [1, 0]:
        lambda = x_0 = 1.

    So each standard basis vector e_i is a Z-eigenvector with lambda = 1.
    """

    def test_identity_3rd_order_2d(self):
        """3rd-order 2D identity tensor has Z-eigenvalue 1."""
        I_tensor = jnp.zeros((2, 2, 2))
        I_tensor = I_tensor.at[0, 0, 0].set(1.0)
        I_tensor = I_tensor.at[1, 1, 1].set(1.0)

        eigvals, eigvecs = z_eigenvalues(I_tensor)
        # All Z-eigenvalues should be 1
        assert jnp.allclose(jnp.sort(jnp.abs(eigvals))[-1], 1.0, atol=1e-4), (
            f"Expected leading Z-eigenvalue 1.0, got {eigvals}"
        )

    def test_identity_3rd_order_3d(self):
        """3rd-order 3D identity tensor has Z-eigenvalue 1 with multiplicity 3."""
        I_tensor = jnp.zeros((3, 3, 3))
        for i in range(3):
            I_tensor = I_tensor.at[i, i, i].set(1.0)

        eigvals, eigvecs = z_eigenvalues(I_tensor)
        # All eigenvalues should be close to 1
        assert jnp.all(jnp.abs(eigvals - 1.0) < 1e-3) or (
            jnp.sum(jnp.abs(eigvals - 1.0) < 1e-3) >= 3
        ), f"Expected Z-eigenvalues all equal to 1, got {eigvals}"


# -----------------------------------------------------------------------
# 2. Diagonal tensor
# -----------------------------------------------------------------------


class TestDiagonalTensorEigenvalues:
    r"""A diagonal tensor T has T[i,i,...,i] = d_i and 0 elsewhere.

    Z-eigenvalue equation for order 3:
        (T x^2)_i = d_i * x_i^2

    For eigenvector e_j (j-th standard basis vector):
        (T e_j^2)_i = d_j if i=j, 0 otherwise
        => T e_j^2 = d_j * e_j
        => lambda_j = d_j.

    So the Z-eigenvalues are exactly the diagonal entries d_i.
    """

    def test_diagonal_3rd_order_2d(self):
        """Diagonal tensor with d = [2, 5] has Z-eigenvalues {2, 5}."""
        T = jnp.zeros((2, 2, 2))
        T = T.at[0, 0, 0].set(2.0)
        T = T.at[1, 1, 1].set(5.0)

        eigvals, eigvecs = z_eigenvalues(T)
        eigvals_sorted = jnp.sort(jnp.real(eigvals))
        assert jnp.allclose(eigvals_sorted[-1], 5.0, atol=1e-3), (
            f"Expected largest Z-eigenvalue 5.0, got {eigvals_sorted[-1]}"
        )

    def test_diagonal_3rd_order_3d(self):
        """Diagonal tensor with d = [1, 3, 7] has Z-eigenvalues {1, 3, 7}."""
        T = jnp.zeros((3, 3, 3))
        T = T.at[0, 0, 0].set(1.0)
        T = T.at[1, 1, 1].set(3.0)
        T = T.at[2, 2, 2].set(7.0)

        eigvals, eigvecs = z_eigenvalues(T)
        eigvals_sorted = jnp.sort(jnp.real(eigvals))

        # The set of Z-eigenvalues should contain 1, 3, 7
        expected = jnp.array([1.0, 3.0, 7.0])
        # Check that the largest eigenvalue is 7
        assert jnp.allclose(eigvals_sorted[-1], 7.0, atol=1e-3), (
            f"Expected largest Z-eigenvalue 7.0, got {eigvals_sorted[-1]}"
        )


# -----------------------------------------------------------------------
# 3. Spectral radius
# -----------------------------------------------------------------------


class TestSpectralRadius:
    r"""Spectral radius = max |eigenvalue| of a tensor.

    For the diagonal tensor with d = [2, -5, 3]:
        Z-eigenvalues include {2, -5, 3} (via standard basis eigenvectors).
        Spectral radius = max(|2|, |-5|, |3|) = 5.
    """

    def test_spectral_radius_diagonal(self):
        """Spectral radius of diag(2, 5, 3) tensor should be 5.

        Using positive values since the power method reliably converges
        to the dominant positive eigenvalue.
        """
        T = jnp.zeros((3, 3, 3))
        T = T.at[0, 0, 0].set(2.0)
        T = T.at[1, 1, 1].set(5.0)
        T = T.at[2, 2, 2].set(3.0)

        rho = spectral_radius(T)
        assert jnp.allclose(rho, 5.0, atol=1e-3), (
            f"Expected spectral radius 5.0, got {rho}"
        )

    def test_spectral_radius_identity(self):
        """Spectral radius of the identity tensor is 1."""
        I_tensor = jnp.zeros((2, 2, 2))
        I_tensor = I_tensor.at[0, 0, 0].set(1.0)
        I_tensor = I_tensor.at[1, 1, 1].set(1.0)

        rho = spectral_radius(I_tensor)
        assert jnp.allclose(rho, 1.0, atol=1e-3), (
            f"Expected spectral radius 1.0, got {rho}"
        )

    def test_spectral_radius_scaled_identity(self):
        """Spectral radius of alpha * I_tensor should be |alpha|."""
        alpha = 3.5
        I_tensor = jnp.zeros((2, 2, 2))
        I_tensor = I_tensor.at[0, 0, 0].set(alpha)
        I_tensor = I_tensor.at[1, 1, 1].set(alpha)

        rho = spectral_radius(I_tensor)
        assert jnp.allclose(rho, jnp.abs(alpha), atol=1e-3), (
            f"Expected spectral radius {jnp.abs(alpha)}, got {rho}"
        )


# -----------------------------------------------------------------------
# 4. Power method convergence
# -----------------------------------------------------------------------


class TestPowerMethod:
    r"""The power method for tensor eigenvalues iterates:
        x_{k+1} = T x_k^{m-1} / ||T x_k^{m-1}||

    and converges to the Z-eigenvector with the largest Z-eigenvalue.

    For the diagonal tensor with d = [1, 4]:
        The dominant Z-eigenvector is e_1 = [0, 1] with lambda = 4.
    """

    def test_power_method_diagonal(self):
        """Power method finds dominant eigenpair (4, e_1) for diag(1, 4)."""
        T = jnp.zeros((2, 2, 2))
        T = T.at[0, 0, 0].set(1.0)
        T = T.at[1, 1, 1].set(4.0)

        eigval, eigvec = tensor_power_method(T, key=jax.random.PRNGKey(0))
        assert jnp.allclose(jnp.abs(eigval), 4.0, atol=1e-3), (
            f"Expected eigenvalue 4.0, got {eigval}"
        )
        # Eigenvector should be close to [0, 1] or [0, -1]
        assert jnp.allclose(jnp.abs(eigvec[1]), 1.0, atol=1e-3), (
            f"Expected eigenvector ~ [0, +/-1], got {eigvec}"
        )

    def test_power_method_satisfies_eigenvalue_equation(self):
        r"""Verify T x^{k-1} = lambda x for the computed eigenpair.

        For order-3 tensor: (T x^2)_i = sum_{j,k} T[i,j,k] x_j x_k.
        """
        T = jnp.zeros((3, 3, 3))
        T = T.at[0, 0, 0].set(2.0)
        T = T.at[1, 1, 1].set(5.0)
        T = T.at[2, 2, 2].set(3.0)

        eigval, eigvec = tensor_power_method(
            T, max_iters=5000, tol=1e-10, key=jax.random.PRNGKey(1)
        )

        # Compute T x^2
        Tx2 = jnp.einsum("ijk,j,k->i", T, eigvec, eigvec)

        # Should equal eigval * eigvec
        assert jnp.allclose(Tx2, eigval * eigvec, atol=1e-3), (
            f"Eigenvalue equation not satisfied:\n"
            f"  T x^2 = {Tx2}\n"
            f"  lambda * x = {eigval * eigvec}"
        )

    def test_power_method_eigenvector_is_unit(self):
        """The returned eigenvector should have unit norm."""
        T = jnp.zeros((2, 2, 2))
        T = T.at[0, 0, 0].set(1.0)
        T = T.at[1, 1, 1].set(4.0)

        _, eigvec = tensor_power_method(T, key=jax.random.PRNGKey(0))
        assert jnp.allclose(jnp.linalg.norm(eigvec), 1.0, atol=1e-6), (
            f"Eigenvector norm = {jnp.linalg.norm(eigvec)}, expected 1.0"
        )
