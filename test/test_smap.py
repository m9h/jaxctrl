"""Known-answer tests for the S-map (sequential locally-weighted linear maps).

The S-map nonlinearity test (Sugihara 1994): the localisation parameter θ controls
neighbour weighting w_i = exp(−θ·d_i/d̄).  θ=0 is a single global linear map
(≡ a linear AR / DMD); if forecast skill *improves* for θ>0 the dynamics are
nonlinear / state-dependent.  So a linear system shows no improvement, a nonlinear
(deterministic) one does.
"""

import numpy as np
import jax.numpy as jnp

from jaxctrl import smap_predict, smap_nonlinearity


def test_smap_linear_no_improvement():
    # AR(1) vector process (linear, Gaussian) -> rho(theta) does not improve for theta>0
    rng = np.random.default_rng(0)
    T, d = 6000, 3
    A = np.array([[0.7, 0.1, 0.0], [0.0, 0.6, 0.1], [0.1, 0.0, 0.5]])
    Z = np.zeros((T, d))
    for t in range(1, T):
        Z[t] = A @ Z[t - 1] + 0.5 * rng.standard_normal(d)
    thetas = np.array([0.0, 0.5, 1.0, 2.0, 4.0, 8.0])
    rho = np.asarray(smap_nonlinearity(jnp.asarray(Z), jnp.asarray(thetas)))
    assert rho[0] > 0.3                                  # the linear map predicts well
    assert rho.max() - rho[0] < 0.05                     # θ>0 gives ~no improvement


def test_smap_nonlinear_improves():
    # logistic map (deterministic nonlinear) -> skill improves with theta
    T = 4000
    x = np.zeros(T)
    x[0] = 0.4
    for t in range(1, T):
        x[t] = 3.9 * x[t - 1] * (1 - x[t - 1])
    # delay embedding (x_t, x_{t-1})
    Z = np.stack([x[1:], x[:-1]], axis=1)
    thetas = np.array([0.0, 0.5, 1.0, 2.0, 4.0, 8.0])
    rho = np.asarray(smap_nonlinearity(jnp.asarray(Z), jnp.asarray(thetas)))
    assert rho.max() - rho[0] > 0.05                     # nonlinear: θ>0 improves skill


def test_smap_predict_shapes_and_exactness():
    # an exactly-linear library -> S-map (any theta) recovers the linear map exactly
    rng = np.random.default_rng(1)
    L, M, d = 200, 30, 4
    B = rng.standard_normal((d, d))
    libX = rng.standard_normal((L, d))
    libY = libX @ B.T
    qX = rng.standard_normal((M, d))
    pred = np.asarray(smap_predict(jnp.asarray(libX), jnp.asarray(libY),
                                   jnp.asarray(qX), 0.0, ridge=1e-8))
    np.testing.assert_allclose(pred, qX @ B.T, atol=1e-3)
