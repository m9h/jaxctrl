"""Known-answer tests for hypergraph control analysis.

These tests require the `hgx` package (optional dependency).  Tests are
skipped automatically when hgx is not installed.

Sections
--------
1. Complete hypergraph adjacency tensor and controllability
2. Path hypergraph and minimum driver nodes
3. Control energy for a driven hypergraph
4. Linear system conversion from hypergraph
"""

import jax
import jax.numpy as jnp
import pytest

try:
    import hgx

    HAS_HGX = True
except ImportError:
    HAS_HGX = False

pytestmark = pytest.mark.skipif(not HAS_HGX, reason="hgx not installed")

from jaxctrl._hypergraph_control import (
    adjacency_tensor,
    control_energy,
    hypergraph_linear_system,
    minimum_driver_nodes,
    tensor_kalman_rank,
)


# -----------------------------------------------------------------------
# 1. Complete hypergraph
# -----------------------------------------------------------------------


class TestCompleteHypergraph:
    r"""A complete hypergraph with one edge containing all n=4 nodes.

    Setup
    -----
    Incidence matrix H = ones(4, 1).
    The single hyperedge {0, 1, 2, 3} connects all nodes.

    Adjacency tensor
    ----------------
    For an order-k uniform hypergraph (here k = 4, since the single edge
    has cardinality 4), the adjacency tensor A has:
        A[i_1, ..., i_k] = 1  if {i_1, ..., i_k} is an edge, 0 otherwise.

    For the complete single-edge hypergraph:
        A[i,j,k,l] = 1 for all permutations of {0,1,2,3} (since the edge
        contains all 4 nodes), and 0 for any tuple with repeated indices
        that does not form a permutation of the edge.

    In the symmetric normalised form, A is symmetric under index permutation.

    Controllability
    ---------------
    With 1 driver node controlling the system, the linearised dynamics
    should be controllable if the hypergraph structure allows influence to
    propagate to all nodes.
    """

    def test_adjacency_tensor_shape(self):
        """Adjacency tensor of 4-node complete hypergraph is 4x4x4x4."""
        H = jnp.ones((4, 1))
        hg = hgx.from_incidence(H)
        A_tensor = adjacency_tensor(hg)
        assert A_tensor.shape == (4, 4, 4, 4), (
            f"Expected shape (4,4,4,4), got {A_tensor.shape}"
        )

    def test_adjacency_tensor_symmetry(self):
        """Adjacency tensor should be symmetric under index permutation."""
        H = jnp.ones((4, 1))
        hg = hgx.from_incidence(H)
        A_tensor = adjacency_tensor(hg)

        # Check a few transpositions
        assert jnp.allclose(
            A_tensor,
            jnp.transpose(A_tensor, (1, 0, 2, 3)),
            atol=1e-6,
        ), "Adjacency tensor not symmetric under swap of indices 0,1"

        assert jnp.allclose(
            A_tensor,
            jnp.transpose(A_tensor, (0, 2, 1, 3)),
            atol=1e-6,
        ), "Adjacency tensor not symmetric under swap of indices 1,2"

    def test_controllability_with_driver_node(self):
        """With node 0 as driver, the complete hypergraph should be controllable.

        All nodes are in the same edge, so influence from node 0 reaches
        all others through the shared hyperedge interaction.
        """
        H = jnp.ones((4, 1))
        hg = hgx.from_incidence(H)
        driver_nodes = jnp.array([0])

        A_t = adjacency_tensor(hg)
        n = hg.num_nodes
        B = jnp.zeros((n, 1)).at[driver_nodes[0], 0].set(1.0)
        rank, is_ctrl = tensor_kalman_rank(A_t, B)
        assert is_ctrl, (
            "Complete hypergraph with driver node 0 should be controllable"
        )


# -----------------------------------------------------------------------
# 2. Path hypergraph
# -----------------------------------------------------------------------


class TestPathHypergraph:
    r"""Path hypergraph: nodes 0-1-2 with edges {0,1} and {1,2}.

    Setup
    -----
    H = [[1, 0],
         [1, 1],
         [0, 1]]

    Minimum driver nodes
    --------------------
    In the pairwise (2-uniform) case, this is a path graph of length 2.
    For a path graph, 1 driver node at either end suffices for
    controllability (it is a connected graph and the controllability
    matrix has full rank).

    Therefore minimum_driver_nodes should return 1.
    """

    def test_minimum_driver_nodes_path(self):
        """Path of 3 nodes needs 1 driver node."""
        H = jnp.array([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        hg = hgx.from_incidence(H)

        n_drivers = minimum_driver_nodes(hg)
        assert n_drivers == 1, (
            f"Expected 1 driver node for path graph, got {n_drivers}"
        )

    def test_disconnected_needs_two_drivers(self):
        """Two disjoint edges {0,1} and {2,3} need at least 2 driver nodes.

        H = [[1,0],[1,0],[0,1],[0,1]]
        Two connected components => minimum 2 drivers.
        """
        H = jnp.array([
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ])
        hg = hgx.from_incidence(H)

        n_drivers = minimum_driver_nodes(hg)
        assert n_drivers >= 2, (
            f"Expected >= 2 driver nodes for disconnected graph, got {n_drivers}"
        )


# -----------------------------------------------------------------------
# 3. Control energy
# -----------------------------------------------------------------------


class TestHypergraphControlEnergy:
    """Control energy for a hypergraph with a known driver node."""

    def test_energy_is_finite(self):
        """For a controllable system, the control energy should be finite."""
        H = jnp.ones((4, 1))
        hg = hgx.from_incidence(H)
        driver_nodes = jnp.array([0])
        x_target = jnp.ones(4)

        E = control_energy(hg, driver_nodes, x_target, T=1.0)
        assert jnp.isfinite(E), f"Energy should be finite, got {E}"

    def test_energy_is_nonnegative(self):
        """Control energy must be non-negative (it is a quadratic form)."""
        H = jnp.array([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        hg = hgx.from_incidence(H)
        driver_nodes = jnp.array([0])
        x_target = jnp.array([1.0, 0.0, 0.0])

        E = control_energy(hg, driver_nodes, x_target, T=1.0)
        assert E >= -1e-8, f"Energy should be >= 0, got {E}"

    def test_zero_target_zero_energy(self):
        """Reaching the origin from the origin requires zero energy."""
        H = jnp.ones((3, 1))
        hg = hgx.from_incidence(H)
        driver_nodes = jnp.array([0])
        x_target = jnp.zeros(3)

        E = control_energy(hg, driver_nodes, x_target, T=1.0)
        assert jnp.allclose(E, 0.0, atol=1e-6), (
            f"Energy to reach origin should be 0, got {E}"
        )


# -----------------------------------------------------------------------
# 4. Linear system conversion
# -----------------------------------------------------------------------


class TestHypergraphLinearSystem:
    r"""Convert a hypergraph to a linear system (A, B).

    For a hypergraph with n nodes and m driver nodes:
        A should be (n, n)
        B should be (n, m)

    The matrix A encodes the hypergraph dynamics and B selects
    which nodes receive external input.
    """

    def test_shapes(self):
        """A is (n, n) and B is (n, m) where m = number of driver nodes."""
        H = jnp.ones((4, 1))
        hg = hgx.from_incidence(H)
        driver_nodes = jnp.array([0, 2])

        A, B = hypergraph_linear_system(hg, driver_nodes)
        assert A.shape == (4, 4), f"Expected A shape (4,4), got {A.shape}"
        assert B.shape == (4, 2), f"Expected B shape (4,2), got {B.shape}"

    def test_B_selects_driver_nodes(self):
        """B should have nonzero entries only in driver-node rows.

        With driver_nodes = [0, 2], B should have nonzeros in rows 0 and 2.
        Rows 1 and 3 should be zero (no direct actuation).
        """
        H = jnp.ones((4, 1))
        hg = hgx.from_incidence(H)
        driver_nodes = jnp.array([0, 2])

        _, B = hypergraph_linear_system(hg, driver_nodes)

        # Driver rows should be nonzero
        assert jnp.any(B[0] != 0), "Driver node 0 row should be nonzero"
        assert jnp.any(B[2] != 0), "Driver node 2 row should be nonzero"
        # Non-driver rows should be zero
        assert jnp.allclose(B[1], 0.0, atol=1e-8), (
            "Non-driver node 1 should have zero row in B"
        )
        assert jnp.allclose(B[3], 0.0, atol=1e-8), (
            "Non-driver node 3 should have zero row in B"
        )

    def test_single_driver(self):
        """Single driver node yields B of shape (n, 1)."""
        H = jnp.array([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        hg = hgx.from_incidence(H)
        driver_nodes = jnp.array([1])

        A, B = hypergraph_linear_system(hg, driver_nodes)
        assert A.shape == (3, 3), f"Expected A shape (3,3), got {A.shape}"
        assert B.shape == (3, 1), f"Expected B shape (3,1), got {B.shape}"

    def test_A_encodes_adjacency(self):
        """For a pairwise path graph, A should encode the graph adjacency.

        H = [[1,0],[1,1],[0,1]] (path 0-1-2).
        The linearised A should have nonzero entries reflecting node
        interactions through shared hyperedges.
        """
        H = jnp.array([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        hg = hgx.from_incidence(H)
        driver_nodes = jnp.array([0])

        A, _ = hypergraph_linear_system(hg, driver_nodes)

        # Nodes 0 and 1 share edge 0 => A[0,1] and A[1,0] should be nonzero
        assert A[0, 1] != 0.0, "A[0,1] should be nonzero (shared edge)"
        assert A[1, 0] != 0.0, "A[1,0] should be nonzero (shared edge)"
        # Nodes 0 and 2 share no edge => A[0,2] should be zero
        assert jnp.allclose(A[0, 2], 0.0, atol=1e-8), (
            "A[0,2] should be zero (no shared edge)"
        )
