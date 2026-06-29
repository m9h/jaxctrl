"""JAX-native Gaussian multivariate transfer entropy + permutation tests.

Gaussian conditional MI = covariance-determinant linear algebra (differentiable,
GPU-vmappable); the multivariate conditioning removes common-driver/cascade confounds
(linear TE ≡ conditional Granger); permutation null via circular shifts, vmap-ed.
"""

import numpy as np
import jax
import jax.numpy as jnp

from jaxctrl import (
    gaussian_cmi,
    transfer_entropy,
    mvte_matrix,
    te_permutation_test,
)


def test_gaussian_cmi_independent_is_zero():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((4000, 1))
    Y = rng.standard_normal((4000, 1))
    Z = rng.standard_normal((4000, 2))
    assert abs(float(gaussian_cmi(jnp.asarray(X), jnp.asarray(Y), jnp.asarray(Z)))) < 0.02


def test_gaussian_cmi_conditional_independence():
    # chain X -> Z -> Y: X,Y dependent but conditionally independent given Z
    rng = np.random.default_rng(1)
    X = rng.standard_normal(6000)
    Z = 0.9 * X + 0.4 * rng.standard_normal(6000)
    Y = 0.9 * Z + 0.4 * rng.standard_normal(6000)
    mi = float(gaussian_cmi(jnp.asarray(X), jnp.asarray(Y)))            # I(X;Y) > 0
    cmi = float(gaussian_cmi(jnp.asarray(X), jnp.asarray(Y), jnp.asarray(Z)))  # | Z -> ~0
    assert mi > 0.3
    assert cmi < 0.05


def test_transfer_entropy_directed():
    # AR chain x -> y : TE(x->y) > 0, TE(y->x) ~ 0 (conditioned on the receiver's past)
    rng = np.random.default_rng(2)
    T = 8000
    x = rng.standard_normal(T)
    y = np.zeros(T)
    for t in range(1, T):
        y[t] = 0.6 * x[t - 1] + 0.4 * rng.standard_normal()
    x, y = jnp.asarray(x), jnp.asarray(y)
    te_xy = float(transfer_entropy(x, y))
    te_yx = float(transfer_entropy(y, x))
    assert te_xy > 0.05
    assert te_yx < 0.2 * te_xy


def test_mvte_removes_common_driver():
    # X drives Y and Z (different lags) -> bivariate TE(Y->Z) spurious; conditional ~0
    rng = np.random.default_rng(3)
    T = 9000
    X = rng.standard_normal(T)
    Y = np.zeros(T)
    Z = np.zeros(T)
    for t in range(2, T):
        Y[t] = 0.7 * X[t - 1] + 0.3 * rng.standard_normal()
        Z[t] = 0.7 * X[t - 2] + 0.3 * rng.standard_normal()
    data = jnp.asarray(np.stack([X, Y, Z], axis=1))                    # (T, 3)
    M = np.asarray(mvte_matrix(data))                                  # conditional TE (n,n)
    # the spurious Y->Z (index 1->2), conditioned on X, is small vs the real X->Y/X->Z
    assert M[1, 2] < 0.3 * M[0, 1]
    assert M[0, 1] > 0.05 and M[0, 2] > 0.05                           # X drives Y and Z


def test_permutation_null_significance():
    rng = np.random.default_rng(4)
    T = 6000
    x = rng.standard_normal(T)
    y = np.zeros(T)
    for t in range(1, T):
        y[t] = 0.6 * x[t - 1] + 0.4 * rng.standard_normal()
    obs, null, z, p = te_permutation_test(jnp.asarray(x), jnp.asarray(y),
                                          n_perm=200, key=jax.random.PRNGKey(0))
    assert z > 3 and p < 0.05                                          # real coupling
    obs2, _, z2, p2 = te_permutation_test(jnp.asarray(y), jnp.asarray(x),
                                          n_perm=200, key=jax.random.PRNGKey(1))
    assert p2 > 0.05                                                  # no reverse coupling
