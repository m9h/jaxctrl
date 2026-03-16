"""Known-answer tests for system identification (SINDy and DMD/Koopman).

Sections
--------
1. Polynomial library construction
2. SINDy on the Lorenz system
3. SINDy linearization
4. Koopman/DMD on a rotation matrix
5. Koopman prediction via eigendecomposition
6. Fourier library construction
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxctrl._sysid import (
    KoopmanEstimator,
    SINDyOptimizer,
    fourier_library,
    polynomial_library,
)


# -----------------------------------------------------------------------
# 1. Polynomial library
# -----------------------------------------------------------------------


class TestPolynomialLibrary:
    """Verify polynomial library construction and column counts."""

    def test_degree_1_columns(self):
        """Degree 1, 2 vars: [1, x1, x2] = 3 columns."""
        X = jnp.ones((5, 2))
        Theta = polynomial_library(X, degree=1)
        assert Theta.shape == (5, 3), f"Expected (5, 3), got {Theta.shape}"

    def test_degree_2_columns(self):
        """Degree 2, 2 vars: [1, x1, x2, x1^2, x1*x2, x2^2] = 6 columns."""
        X = jnp.ones((5, 2))
        Theta = polynomial_library(X, degree=2)
        assert Theta.shape == (5, 6), f"Expected (5, 6), got {Theta.shape}"

    def test_degree_2_3vars(self):
        """Degree 2, 3 vars: 1 + 3 + 6 = 10 columns."""
        X = jnp.ones((5, 3))
        Theta = polynomial_library(X, degree=2)
        assert Theta.shape == (5, 10), f"Expected (5, 10), got {Theta.shape}"

    def test_constant_column_is_ones(self):
        """First column should always be 1."""
        X = jax.random.normal(jax.random.PRNGKey(0), (10, 3))
        Theta = polynomial_library(X, degree=2)
        assert jnp.allclose(Theta[:, 0], 1.0)

    def test_linear_columns_match_X(self):
        """Columns 1:n_vars+1 should equal X for degree >= 1."""
        X = jax.random.normal(jax.random.PRNGKey(0), (10, 3))
        Theta = polynomial_library(X, degree=2)
        assert jnp.allclose(Theta[:, 1:4], X)

    def test_quadratic_terms(self):
        """For 2 vars, degree 2: col 3 = x1^2, col 4 = x1*x2, col 5 = x2^2."""
        X = jnp.array([[2.0, 3.0], [1.0, 4.0]])
        Theta = polynomial_library(X, degree=2)
        assert jnp.allclose(Theta[:, 3], X[:, 0] ** 2)
        assert jnp.allclose(Theta[:, 4], X[:, 0] * X[:, 1])
        assert jnp.allclose(Theta[:, 5], X[:, 1] ** 2)


# -----------------------------------------------------------------------
# 2. SINDy on the Lorenz system
# -----------------------------------------------------------------------


class TestSINDyLorenz:
    r"""Recover the Lorenz system from simulated data.

    The Lorenz system:
        dx/dt = sigma * (y - x)
        dy/dt = x * (rho - z) - y
        dz/dt = x * y - beta * z

    With sigma=10, rho=28, beta=8/3.

    In polynomial library form (degree 2, 3 vars):
        Theta = [1, x, y, z, x^2, xy, xz, y^2, yz, z^2]
        Xi columns correspond to dx, dy, dz.

    Expected nonzero coefficients:
        dx: y -> +10, x -> -10
        dy: x -> +28, y -> -1, xz -> -1
        dz: xy -> +1, z -> -8/3
    """

    @pytest.fixture
    def lorenz_data(self):
        """Generate Lorenz trajectory via RK4 integration."""
        import numpy as np

        sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0
        dt = 0.001
        n_total = 30000
        n_transient = 5000  # discard initial transient

        def lorenz_rhs(state):
            x, y, z = state
            return np.array([
                sigma * (y - x),
                x * (rho - z) - y,
                x * y - beta * z,
            ])

        # RK4 integration in float64
        state = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        states, derivs = [], []
        for i in range(n_total):
            k1 = lorenz_rhs(state)
            k2 = lorenz_rhs(state + 0.5 * dt * k1)
            k3 = lorenz_rhs(state + 0.5 * dt * k2)
            k4 = lorenz_rhs(state + dt * k3)
            if i >= n_transient:
                states.append(state.copy())
                derivs.append(k1.copy())
            state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        X = jnp.array(np.stack(states))
        dX = jnp.array(np.stack(derivs))
        return X, dX

    def test_lorenz_recovery(self, lorenz_data):
        """SINDy recovers Lorenz coefficients from clean data."""
        X, dX = lorenz_data
        lib_fn = lambda x: polynomial_library(x, degree=2)

        opt = SINDyOptimizer(threshold=0.5, max_iter=10)
        Xi = opt.fit(X, dX, lib_fn)

        # Column indices in degree-2, 3-var library:
        # 0: const, 1: x, 2: y, 3: z,
        # 4: x^2, 5: xy, 6: xz, 7: y^2, 8: yz, 9: z^2

        # dx/dt = -10x + 10y
        assert jnp.abs(Xi[1, 0] - (-10.0)) < 1.0, f"dx/x coeff: {Xi[1, 0]}"
        assert jnp.abs(Xi[2, 0] - 10.0) < 1.0, f"dx/y coeff: {Xi[2, 0]}"

        # dy/dt = 28x - y - xz
        assert jnp.abs(Xi[1, 1] - 28.0) < 1.0, f"dy/x coeff: {Xi[1, 1]}"
        assert jnp.abs(Xi[2, 1] - (-1.0)) < 0.5, f"dy/y coeff: {Xi[2, 1]}"
        assert jnp.abs(Xi[6, 1] - (-1.0)) < 0.5, f"dy/xz coeff: {Xi[6, 1]}"

        # dz/dt = xy - (8/3)z
        assert jnp.abs(Xi[5, 2] - 1.0) < 0.5, f"dz/xy coeff: {Xi[5, 2]}"
        assert jnp.abs(Xi[3, 2] - (-8.0 / 3.0)) < 0.5, f"dz/z coeff: {Xi[3, 2]}"

    def test_sparsity(self, lorenz_data):
        """Recovered Xi should be sparse (< 10 nonzero entries)."""
        X, dX = lorenz_data
        lib_fn = lambda x: polynomial_library(x, degree=2)

        opt = SINDyOptimizer(threshold=0.5, max_iter=10)
        Xi = opt.fit(X, dX, lib_fn)

        n_nonzero = jnp.sum(jnp.abs(Xi) > 0.01)
        assert n_nonzero <= 10, f"Expected sparse Xi, got {n_nonzero} nonzero entries"

    def test_predict(self, lorenz_data):
        """Predicted derivatives should match true derivatives."""
        X, dX = lorenz_data
        lib_fn = lambda x: polynomial_library(x, degree=2)

        opt = SINDyOptimizer(threshold=0.5, max_iter=10)
        Xi = opt.fit(X, dX, lib_fn)
        dX_pred = opt.predict(X, Xi, lib_fn)

        mse = jnp.mean((dX - dX_pred) ** 2)
        assert mse < 1.0, f"Prediction MSE = {mse}"


# -----------------------------------------------------------------------
# 3. SINDy linearization
# -----------------------------------------------------------------------


class TestSINDyLinearize:
    """Extract A matrix from SINDy for control analysis."""

    def test_linear_system_recovery(self):
        r"""For a linear system dX/dt = A @ X, SINDy with degree 1
        should recover A exactly, and linearize() should return it.

        A = [[-0.5, 1.0], [0.0, -1.0]]
        """
        A_true = jnp.array([[-0.5, 1.0], [0.0, -1.0]])

        # Generate data
        key = jax.random.PRNGKey(0)
        X = jax.random.normal(key, (500, 2))
        dX = (A_true @ X.T).T

        lib_fn = lambda x: polynomial_library(x, degree=1)
        opt = SINDyOptimizer(threshold=0.01, max_iter=5)
        Xi = opt.fit(X, dX, lib_fn)

        A_recovered = SINDyOptimizer.linearize(Xi, n_vars=2)
        assert jnp.allclose(A_recovered, A_true, atol=1e-3), (
            f"Expected:\n{A_true}\nGot:\n{A_recovered}"
        )


# -----------------------------------------------------------------------
# 4. Koopman/DMD on a rotation matrix
# -----------------------------------------------------------------------


class TestKoopmanRotation:
    r"""Exact DMD on a pure rotation should recover the rotation matrix.

    K = [[cos(theta), -sin(theta)],
         [sin(theta),  cos(theta)]]

    with theta = pi/4.

    Eigenvalues should be exp(+/- i * theta) with |lambda| = 1.
    """

    @pytest.fixture
    def rotation_data(self):
        theta = jnp.pi / 4
        K_true = jnp.array([
            [jnp.cos(theta), -jnp.sin(theta)],
            [jnp.sin(theta), jnp.cos(theta)],
        ])
        x0 = jnp.array([1.0, 0.0])

        n_steps = 100
        snapshots = [x0]
        x = x0
        for _ in range(n_steps):
            x = K_true @ x
            snapshots.append(x)

        data = jnp.stack(snapshots).T  # (2, n_steps+1)
        X = data[:, :-1]
        Y = data[:, 1:]
        return X, Y, K_true

    def test_operator_recovery(self, rotation_data):
        """K_est should match K_true."""
        X, Y, K_true = rotation_data
        est = KoopmanEstimator()
        K_est, _, _ = est.fit(X, Y)

        assert jnp.allclose(K_est, K_true, atol=1e-5), (
            f"Expected:\n{K_true}\nGot:\n{K_est}"
        )

    def test_eigenvalues_on_unit_circle(self, rotation_data):
        """DMD eigenvalues should have |lambda| = 1 for rotation."""
        X, Y, _ = rotation_data
        est = KoopmanEstimator()
        _, eigenvalues, _ = est.fit(X, Y)

        mags = jnp.abs(eigenvalues)
        assert jnp.allclose(mags, 1.0, atol=1e-5), (
            f"Eigenvalue magnitudes: {mags}"
        )

    def test_eigenvalue_frequency(self, rotation_data):
        """Eigenvalue angles should be +/- pi/4."""
        X, Y, _ = rotation_data
        est = KoopmanEstimator()
        _, eigenvalues, _ = est.fit(X, Y)

        angles = jnp.sort(jnp.angle(eigenvalues))
        expected = jnp.sort(jnp.array([-jnp.pi / 4, jnp.pi / 4]))
        assert jnp.allclose(angles, expected, atol=1e-5), (
            f"Expected angles {expected}, got {angles}"
        )


# -----------------------------------------------------------------------
# 5. Koopman prediction
# -----------------------------------------------------------------------


class TestKoopmanPrediction:
    """Test eigendecomposition-based prediction vs ground truth."""

    def test_prediction_rotation(self):
        """Predict 10 steps of a rotation and compare to ground truth."""
        theta = jnp.pi / 6
        K_true = jnp.array([
            [jnp.cos(theta), -jnp.sin(theta)],
            [jnp.sin(theta), jnp.cos(theta)],
        ])
        x0 = jnp.array([1.0, 0.0])

        # Generate training data
        snapshots = [x0]
        x = x0
        for _ in range(50):
            x = K_true @ x
            snapshots.append(x)
        data = jnp.stack(snapshots).T
        X, Y = data[:, :-1], data[:, 1:]

        est = KoopmanEstimator()
        _, eigenvalues, modes = est.fit(X, Y)

        # Predict at t=10
        x_pred = est.predict(x0, 10, eigenvalues, modes)
        x_true = jnp.linalg.matrix_power(K_true, 10) @ x0

        assert jnp.allclose(x_pred, x_true, atol=1e-4), (
            f"Predicted: {x_pred}, True: {x_true}"
        )

    def test_continuous_eigenvalues(self):
        """Continuous eigenvalues of a rotation = +/- i*theta/dt."""
        theta = jnp.pi / 4
        dt = 0.01
        K = jnp.array([
            [jnp.cos(theta), -jnp.sin(theta)],
            [jnp.sin(theta), jnp.cos(theta)],
        ])

        eigenvalues = jnp.linalg.eigvals(K)
        omega = KoopmanEstimator.continuous_eigenvalues(eigenvalues, dt)

        # Real parts should be ~0 (no growth/decay)
        assert jnp.allclose(jnp.real(omega), 0.0, atol=1e-5), (
            f"Real parts: {jnp.real(omega)}"
        )
        # Imaginary parts should be +/- theta/dt
        expected_imag = jnp.sort(jnp.array([-theta / dt, theta / dt]))
        assert jnp.allclose(jnp.sort(jnp.imag(omega)), expected_imag, atol=1e-3)

    def test_stability_check(self):
        """Stable system has |eigenvalues| < 1."""
        # Damped rotation
        eigs = jnp.array([0.9 + 0.1j, 0.9 - 0.1j])
        assert KoopmanEstimator.is_stable(eigs)

        # Unstable
        eigs_bad = jnp.array([1.1 + 0.0j, 0.5 + 0.0j])
        assert not KoopmanEstimator.is_stable(eigs_bad)


# -----------------------------------------------------------------------
# 6. Fourier library
# -----------------------------------------------------------------------


class TestFourierLibrary:
    """Test Fourier feature library construction."""

    def test_column_count(self):
        """n_freqs=2, 3 vars: 1 + 2*3*2 = 13 columns."""
        X = jnp.ones((5, 3))
        Theta = fourier_library(X, n_freqs=2)
        assert Theta.shape == (5, 13), f"Expected (5, 13), got {Theta.shape}"

    def test_constant_column(self):
        """First column is always 1."""
        X = jax.random.normal(jax.random.PRNGKey(0), (10, 2))
        Theta = fourier_library(X, n_freqs=3)
        assert jnp.allclose(Theta[:, 0], 1.0)

    def test_cos_sin_pairs(self):
        """Columns alternate cos and sin for each variable."""
        X = jnp.array([[0.5, 1.0]])
        Theta = fourier_library(X, n_freqs=1)
        # Columns: [1, cos(x1), sin(x1), cos(x2), sin(x2)]
        assert jnp.allclose(Theta[0, 1], jnp.cos(0.5))
        assert jnp.allclose(Theta[0, 2], jnp.sin(0.5))
        assert jnp.allclose(Theta[0, 3], jnp.cos(1.0))
        assert jnp.allclose(Theta[0, 4], jnp.sin(1.0))
