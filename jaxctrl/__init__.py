"""jaxctrl: Differentiable control theory in JAX.

Four-layer architecture:

Layer 0 -- System identification (SINDy, DMD/Koopman)
Layer 1 -- Control primitives (Lyapunov, Riccati, Gramians, controllability)
Layer 2 -- Tensor algebra + control (decompositions, eigenvalues, ARTE)
Layer 3 -- Hypergraph control (integrates with hgx for higher-order networks)

Built on the Kidger stack: Equinox, Lineax, Optimistix, Diffrax.
"""

from __future__ import annotations

import importlib.metadata

# Layer 0: System identification
from jaxctrl._sysid import (
    KoopmanEstimator,
    SINDyOptimizer,
    fourier_library,
    polynomial_library,
)

# Layer 1: Control-theoretic matrix equation solvers
from jaxctrl._lyapunov import (
    is_schur_stable,
    is_stable,
    solve_continuous_lyapunov,
    solve_discrete_lyapunov,
)
from jaxctrl._riccati import (
    dlqr,
    lqr,
    solve_continuous_are,
    solve_discrete_are,
)
from jaxctrl._gramian import (
    controllability_gramian,
    controllability_matrix,
    observability_gramian,
    observability_matrix,
)
from jaxctrl._controllability import (
    is_controllable,
    is_detectable,
    is_observable,
    is_stabilizable,
    minimum_energy,
)
from jaxctrl._simulate import (
    simulate_closed_loop,
    simulate_lti,
)

# Layer 2: Tensor algebra and control primitives
from jaxctrl._tensor_ops import (
    einstein_product,
    hosvd,
    khatri_rao,
    mode_dot,
    symmetrize_tensor,
    tensor_contract,
    tensor_fold,
    tensor_trace,
    tensor_unfold,
    tucker_to_tensor,
)
from jaxctrl._tensor_eigen import (
    h_eigenvalues,
    spectral_radius,
    tensor_power_method,
    z_eigenvalues,
)
from jaxctrl._arte import (
    multilinear_lqr,
    solve_arte,
    tensor_lyapunov,
)

# Layer 3: Hypergraph control (optional, requires hgx)
try:
    from jaxctrl._hypergraph_control import (
        HypergraphControlSystem,
        adjacency_tensor,
        control_energy,
        controllability_profile,
        hypergraph_linear_system,
        laplacian_tensor,
        minimum_driver_nodes,
        tensor_kalman_rank,
    )
except ImportError:
    pass

__version__ = importlib.metadata.version("jaxctrl")
