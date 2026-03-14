"""Known-answer tests for tensor operations.

Every test computes an expected result by hand (documented in the docstring)
and compares it to the implementation output.

Sections
--------
1. Unfold/fold roundtrip
2. Tensor contraction (n-mode product)
3. Einstein product
4. Symmetrisation under index permutation
5. Khatri-Rao product
"""

import jax
import jax.numpy as jnp
import pytest
from jaxctrl._tensor_ops import (
    einstein_product,
    khatri_rao,
    symmetrize_tensor,
    tensor_contract,
    tensor_fold,
    tensor_unfold,
)


# -----------------------------------------------------------------------
# 1. Unfold/fold roundtrip
# -----------------------------------------------------------------------


class TestUnfoldFold:
    r"""Verify that unfolding and folding are exact inverses.

    Mode-k unfolding of a tensor T \in R^{I_1 x ... x I_N} rearranges it
    into a matrix T_(k) \in R^{I_k x (I_1 ... I_{k-1} I_{k+1} ... I_N)}
    by placing the k-th mode as rows and collapsing other modes into columns
    in the standard (Kolda & Bader) ordering.

    Roundtrip: fold(unfold(T, k), k, shape) == T.
    """

    def test_unfold_fold_mode0(self):
        """Unfold mode-0 of a 3x3x3 tensor, then fold back."""
        T = jnp.arange(27.0).reshape(3, 3, 3)
        T_unf = tensor_unfold(T, mode=0)
        T_rec = tensor_fold(T_unf, mode=0, shape=(3, 3, 3))
        assert jnp.allclose(T_rec, T, atol=1e-10), "Roundtrip failed for mode-0"

    def test_unfold_fold_mode1(self):
        """Unfold mode-1 of a 3x3x3 tensor, then fold back."""
        T = jnp.arange(27.0).reshape(3, 3, 3)
        T_unf = tensor_unfold(T, mode=1)
        T_rec = tensor_fold(T_unf, mode=1, shape=(3, 3, 3))
        assert jnp.allclose(T_rec, T, atol=1e-10), "Roundtrip failed for mode-1"

    def test_unfold_fold_mode2(self):
        """Unfold mode-2 of a 3x3x3 tensor, then fold back."""
        T = jnp.arange(27.0).reshape(3, 3, 3)
        T_unf = tensor_unfold(T, mode=2)
        T_rec = tensor_fold(T_unf, mode=2, shape=(3, 3, 3))
        assert jnp.allclose(T_rec, T, atol=1e-10), "Roundtrip failed for mode-2"

    def test_unfold_shape_mode0(self):
        """Mode-0 unfolding of (2, 3, 4) tensor gives (2, 12) matrix."""
        T = jnp.arange(24.0).reshape(2, 3, 4)
        T_unf = tensor_unfold(T, mode=0)
        assert T_unf.shape == (2, 12), f"Expected (2, 12), got {T_unf.shape}"

    def test_unfold_shape_mode1(self):
        """Mode-1 unfolding of (2, 3, 4) tensor gives (3, 8) matrix."""
        T = jnp.arange(24.0).reshape(2, 3, 4)
        T_unf = tensor_unfold(T, mode=1)
        assert T_unf.shape == (3, 8), f"Expected (3, 8), got {T_unf.shape}"

    def test_unfold_known_values_2x2x2(self):
        r"""Mode-0 unfolding of a 2x2x2 identity-like tensor.

        T[i,j,k] = 1 if i==j==k else 0.
        T[:,:,0] = [[1,0],[0,0]], T[:,:,1] = [[0,0],[0,1]]

        Mode-0 unfolding (Kolda convention): rows = mode-0 index,
        columns cycle through (mode-1, mode-2) in order.
        T_(0) = [[T[0,0,0], T[0,1,0], T[0,0,1], T[0,1,1]],
                  [T[1,0,0], T[1,1,0], T[1,0,1], T[1,1,1]]]
               = [[1, 0, 0, 0],
                  [0, 0, 0, 1]]
        """
        T = jnp.zeros((2, 2, 2))
        T = T.at[0, 0, 0].set(1.0)
        T = T.at[1, 1, 1].set(1.0)

        T_unf = tensor_unfold(T, mode=0)
        expected = jnp.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
        assert jnp.allclose(T_unf, expected, atol=1e-10), (
            f"Expected:\n{expected}\nGot:\n{T_unf}"
        )

    def test_roundtrip_nonsquare(self):
        """Roundtrip on a (2, 3, 4) tensor for all modes."""
        T = jnp.arange(24.0).reshape(2, 3, 4)
        for mode in range(3):
            T_unf = tensor_unfold(T, mode=mode)
            T_rec = tensor_fold(T_unf, mode=mode, shape=(2, 3, 4))
            assert jnp.allclose(T_rec, T, atol=1e-10), (
                f"Roundtrip failed for mode-{mode}"
            )


# -----------------------------------------------------------------------
# 2. Tensor contraction (n-mode product)
# -----------------------------------------------------------------------


class TestTensorContraction:
    r"""n-mode product T x_k M multiplies T along mode k by matrix M.

    For a 3rd-order tensor A and a vector x, verify:
        A x_1 x x_2 x = sum over einsum.

    Specifically: for T in R^{2x2x2} and x in R^2:
        (T x_0 x^T) x_1 x^T  contracts modes 0 and 1 with x,
        yielding a vector in R^2.

    This should match jnp.einsum('ijk,i,j->k', T, x, x).
    """

    def test_contract_vector_mode0(self):
        r"""Contract mode-0 of a 2x3x4 tensor with a length-2 vector.

        Result shape: (3, 4).
        Manually: result[j, k] = sum_i T[i, j, k] * v[i].
        """
        T = jnp.arange(24.0).reshape(2, 3, 4)
        v = jnp.array([1.0, 2.0])
        result = tensor_contract(T, v, modes=(0,))
        expected = jnp.einsum("ijk,i->jk", T, v)
        assert jnp.allclose(result, expected, atol=1e-6)

    def test_double_contraction_matches_einsum(self):
        r"""For T(2,2,2) and x(2), T x_0 x x_1 x matches einsum('ijk,i,j->k').

        T = [[[1,2],[3,4]], [[5,6],[7,8]]]
        x = [1, 1]

        einsum('ijk,i,j->k', T, x, x):
            k=0: T[0,0,0]*1*1 + T[0,1,0]*1*1 + T[1,0,0]*1*1 + T[1,1,0]*1*1
                = 1 + 3 + 5 + 7 = 16
            k=1: 2 + 4 + 6 + 8 = 20
        """
        T = jnp.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
        x = jnp.array([1.0, 1.0])

        # Contract mode 0 first, then mode 0 of the result (originally mode 1)
        result_1 = tensor_contract(T, x, modes=(0,))  # shape (2, 2)
        result_2 = tensor_contract(result_1, x, modes=(0,))  # shape (2,)

        expected = jnp.einsum("ijk,i,j->k", T, x, x)
        assert jnp.allclose(result_2, expected, atol=1e-6), (
            f"Expected {expected}, got {result_2}"
        )

    def test_contraction_multi_mode(self):
        r"""Contract modes (0, 1) of a (3,3,3) tensor with vector (3,).

        result[k] = sum_{i,j} T[i,j,k] * v[i] * v[j].
        """
        T = jnp.arange(27.0).reshape(3, 3, 3)
        v = jnp.array([1.0, 0.0, 0.0])
        result = tensor_contract(T, v, modes=(0, 1))
        expected = jnp.einsum("ijk,i,j->k", T, v, v)
        assert jnp.allclose(result, expected, atol=1e-6)


# -----------------------------------------------------------------------
# 3. Einstein product
# -----------------------------------------------------------------------


class TestEinsteinProduct:
    r"""The Einstein product A *_p B for tensors A in R^{I_1 x...x I_N}
    and B in R^{J_1 x...x J_M} contracts the last p indices of A with
    the first p indices of B (they must match).

    For matrices (order-2 tensors), A *_1 B is standard matrix multiplication.
    """

    def test_matrix_multiplication(self):
        """Einstein product with p=1 on 2D tensors is matrix multiplication."""
        A = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        B = jnp.array([[5.0, 6.0], [7.0, 8.0]])
        result = einstein_product(A, B, modes=1)
        expected = A @ B
        assert jnp.allclose(result, expected, atol=1e-6)

    def test_tensor_matrix_product(self):
        r"""A in R^{2x3x4}, B in R^{4x5}, p=1.

        Result in R^{2x3x5}.
        result[i,j,k] = sum_l A[i,j,l] * B[l,k].
        """
        A = jnp.arange(24.0).reshape(2, 3, 4)
        B = jnp.arange(20.0).reshape(4, 5)
        result = einstein_product(A, B, modes=1)
        expected = jnp.einsum("ijl,lk->ijk", A, B)
        assert jnp.allclose(result, expected, atol=1e-5)

    def test_full_contraction(self):
        r"""Full contraction of two (2,3) tensors with p=2 gives a scalar.

        result = sum_{i,j} A[i,j] * B[i,j]  (Frobenius inner product).
        """
        A = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        B = jnp.array([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])
        result = einstein_product(A, B, modes=2)
        expected = jnp.sum(A * B)
        assert jnp.allclose(result, expected, atol=1e-6)


# -----------------------------------------------------------------------
# 4. Symmetrisation
# -----------------------------------------------------------------------


class TestSymmetriseTensor:
    r"""Symmetrise a tensor so it is invariant under all index permutations.

    For an order-k tensor T in R^{n x n x ... x n}:
        sym(T)[i_1, ..., i_k] = (1/k!) sum_{sigma in S_k} T[i_{sigma(1)}, ..., i_{sigma(k)}]
    """

    def test_symmetric_matrix(self):
        """Symmetrisation of a 2x2 matrix is (A + A^T) / 2."""
        A = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        result = symmetrize_tensor(A)
        expected = (A + A.T) / 2.0
        assert jnp.allclose(result, expected, atol=1e-10)

    def test_already_symmetric(self):
        """Symmetrising a symmetric tensor is a no-op."""
        T = jnp.zeros((2, 2, 2))
        T = T.at[0, 0, 0].set(1.0)
        T = T.at[1, 1, 1].set(1.0)
        result = symmetrize_tensor(T)
        assert jnp.allclose(result, T, atol=1e-10)

    def test_symmetrised_is_permutation_invariant(self):
        """After symmetrisation, T[i,j,k] = T[j,i,k] = T[k,j,i] etc."""
        T = jnp.arange(8.0).reshape(2, 2, 2)
        S = symmetrize_tensor(T)

        # Check all transpositions
        assert jnp.allclose(S, jnp.transpose(S, (1, 0, 2)), atol=1e-10)
        assert jnp.allclose(S, jnp.transpose(S, (0, 2, 1)), atol=1e-10)
        assert jnp.allclose(S, jnp.transpose(S, (2, 1, 0)), atol=1e-10)

    def test_3rd_order_known_value(self):
        r"""Symmetrise T with T[0,0,1]=6, all others 0.

        There are 3! = 6 permutations.  T[0,0,1]=6 maps to:
            (0,0,1), (0,1,0), (1,0,0)  => 3 distinct positions
        Each of these 3 positions appears in 2 permutations (since two
        indices are equal), so:
            sym(T)[0,0,1] = sym(T)[0,1,0] = sym(T)[1,0,0] = 6 * 2 / 6 = 2.
        All other entries remain 0.
        """
        T = jnp.zeros((2, 2, 2))
        T = T.at[0, 0, 1].set(6.0)
        S = symmetrize_tensor(T)
        assert jnp.allclose(S[0, 0, 1], 2.0, atol=1e-10)
        assert jnp.allclose(S[0, 1, 0], 2.0, atol=1e-10)
        assert jnp.allclose(S[1, 0, 0], 2.0, atol=1e-10)
        assert jnp.allclose(S[0, 0, 0], 0.0, atol=1e-10)
        assert jnp.allclose(S[1, 1, 1], 0.0, atol=1e-10)


# -----------------------------------------------------------------------
# 5. Khatri-Rao product
# -----------------------------------------------------------------------


class TestKhatriRao:
    r"""Khatri-Rao product: column-wise Kronecker product.

    Given A in R^{I x K} and B in R^{J x K}:
        (A ⊙ B)[:, k] = A[:, k] ⊗ B[:, k]

    Result shape: (I*J, K).
    """

    def test_khatri_rao_simple(self):
        r"""A = [[1, 2], [3, 4]], B = [[5, 6], [7, 8]].

        Column 0: [1,3] ⊗ [5,7] = [5, 7, 15, 21]
        Column 1: [2,4] ⊗ [6,8] = [12, 16, 24, 32]

        Result: [[5, 12], [7, 16], [15, 24], [21, 32]]
        """
        A = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        B = jnp.array([[5.0, 6.0], [7.0, 8.0]])
        result = khatri_rao(A, B)
        expected = jnp.array([
            [5.0, 12.0],
            [7.0, 16.0],
            [15.0, 24.0],
            [21.0, 32.0],
        ])
        assert jnp.allclose(result, expected, atol=1e-6), (
            f"Expected:\n{expected}\nGot:\n{result}"
        )

    def test_khatri_rao_shape(self):
        """(3, 4) and (5, 4) -> (15, 4)."""
        A = jnp.ones((3, 4))
        B = jnp.ones((5, 4))
        result = khatri_rao(A, B)
        assert result.shape == (15, 4), f"Expected (15, 4), got {result.shape}"

    def test_khatri_rao_identity_columns(self):
        r"""When A and B each have one column, KR is the Kronecker product.

        A = [[1], [2]], B = [[3], [4]]
        A ⊙ B = A ⊗ B = [[3], [4], [6], [8]]
        """
        A = jnp.array([[1.0], [2.0]])
        B = jnp.array([[3.0], [4.0]])
        result = khatri_rao(A, B)
        expected = jnp.array([[3.0], [4.0], [6.0], [8.0]])
        assert jnp.allclose(result, expected, atol=1e-10)

    def test_khatri_rao_vs_manual(self, key):
        """Verify against a manual column-by-column Kronecker computation."""
        k1, k2 = jax.random.split(key)
        A = jax.random.normal(k1, (3, 5))
        B = jax.random.normal(k2, (4, 5))

        result = khatri_rao(A, B)

        # Manual: column-wise Kronecker
        cols = []
        for k in range(5):
            cols.append(jnp.kron(A[:, k], B[:, k]).reshape(-1, 1))
        expected = jnp.concatenate(cols, axis=1)

        assert jnp.allclose(result, expected, atol=1e-5), (
            f"Khatri-Rao does not match manual column-wise Kronecker"
        )
