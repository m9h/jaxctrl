"""Control-theoretic analysis of hypergraph dynamical systems.

This module provides tools for analysing controllability, observability, and
control energy of dynamical systems defined on hypergraphs.  It integrates with
the ``hgx`` library for hypergraph representations and follows the Kidger-stack
pattern: all state lives in ``equinox.Module`` leaves and every public function
is pure-functional and JIT-compatible.

**Background.**  A *k*-uniform hypergraph on *n* nodes defines a *k*-th order
tensor adjacency structure.  By *unfolding* (matricising) this tensor we recover
a standard matrix dynamical system and can apply classical Kalman-rank tests,
Gramian-based energy computations, and LQR design.

Requires the optional ``hgx`` dependency (install via
``pip install jaxctrl[hypergraph]``).

References
----------
Chen, C., et al. "Tensor-based controllability analysis of hypergraph
dynamical systems." *IEEE Trans. Automatic Control*, 2023.
"""

from __future__ import annotations

import warnings
from functools import partial
from typing import Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import lax

from jaxctrl._tensor_ops import tensor_unfold

# ---------------------------------------------------------------------------
# Optional hgx import
# ---------------------------------------------------------------------------

try:
    import hgx  # type: ignore[import-untyped]

    HAS_HGX = True
except ImportError:
    hgx = None  # type: ignore[assignment]
    HAS_HGX = False


def _require_hgx(fn_name: str) -> None:
    """Raise a helpful ``ImportError`` if hgx is not installed."""
    if not HAS_HGX:
        raise ImportError(
            f"{fn_name} requires the 'hgx' package.  "
            "Install it with:  pip install jaxctrl[hypergraph]"
        )


# ===================================================================
# 1. Adjacency tensor
# ===================================================================


def adjacency_tensor(
    hg,
    order: Optional[int] = None,
) -> jax.Array:
    """Construct the *k*-th order adjacency tensor from an hgx Hypergraph.

    For each hyperedge *e* = {v1, ..., vk} the entry A[v1, v2, ..., vk] is
    set to 1.  All other entries are 0.

    Parameters
    ----------
    hg : hgx.Hypergraph
        Hypergraph object with ``.incidence``, ``.num_nodes``, ``.num_edges``.
    order : int, optional
        Tensor order (number of indices).  If *None*, the maximum hyperedge
        cardinality is used.  All hyperedges must have exactly this cardinality;
        if not, a ``ValueError`` is raised suggesting uniform decomposition.

    Returns
    -------
    A : jax.Array, shape (n, n, ..., n) with *order* indices
        The adjacency tensor.

    Warns
    -----
    UserWarning
        If ``hg.num_nodes > 100``, since the dense tensor can be very large.
    """
    _require_hgx("adjacency_tensor")

    H = jnp.asarray(hg.incidence)  # (n_nodes, n_edges)
    n = hg.num_nodes
    n_edges = hg.num_edges

    if n > 100:
        warnings.warn(
            f"adjacency_tensor: n={n} > 100. The dense tensor will have "
            f"n^k entries and may consume excessive memory.",
            stacklevel=2,
        )

    # Determine edge cardinalities from the incidence matrix.
    cardinalities = jnp.sum(H, axis=0).astype(jnp.int32)  # (n_edges,)
    max_card = int(jnp.max(cardinalities))

    if order is None:
        order = max_card

    # Validate uniformity: every edge must have cardinality == order.
    min_card = int(jnp.min(cardinalities))
    if min_card != max_card or max_card != order:
        raise ValueError(
            f"Non-uniform hypergraph detected (edge cardinalities range from "
            f"{min_card} to {max_card}, requested order={order}).  "
            "Use a uniform decomposition before calling adjacency_tensor."
        )

    # Build the tensor by iterating over edges.  We extract nodes per edge
    # from the incidence matrix.
    shape = tuple([n] * order)
    A = jnp.zeros(shape, dtype=H.dtype)

    # Each column of H is an edge indicator.  We need the node indices of
    # each edge.  Since we are outside JIT here (dense tensor construction
    # is inherently non-JIT-friendly), we use NumPy-level indexing.
    import itertools

    H_np = jnp.asarray(H)
    for e_idx in range(n_edges):
        col = H_np[:, e_idx]
        nodes = jnp.where(col > 0, size=order)[0]
        # Set all permutations of the node tuple to 1.
        for perm in itertools.permutations(range(order)):
            idx = tuple(int(nodes[p]) for p in perm)
            A = A.at[idx].set(1.0)

    return A


# ===================================================================
# 2. Laplacian tensor
# ===================================================================


def laplacian_tensor(
    hg,
    order: Optional[int] = None,
) -> jax.Array:
    """Compute the tensor Laplacian of a hypergraph.

    The Laplacian tensor is defined as L = D - A, where A is the adjacency
    tensor and D is the *degree tensor* — a diagonal-like tensor whose only
    nonzero entries are the super-diagonal entries D[i, i, ..., i] = deg(i).

    Parameters
    ----------
    hg : hgx.Hypergraph
        Hypergraph object.
    order : int, optional
        Tensor order (see :func:`adjacency_tensor`).

    Returns
    -------
    L : jax.Array, shape (n, n, ..., n)
        Laplacian tensor.
    """
    _require_hgx("laplacian_tensor")

    A = adjacency_tensor(hg, order=order)
    k = A.ndim
    n = A.shape[0]

    # Degree of node i = number of hyperedges containing i.
    H = jnp.asarray(hg.incidence)
    degrees = jnp.sum(H, axis=1)  # (n,)

    # Build degree tensor: D[i, i, ..., i] = deg(i), all others 0.
    D = jnp.zeros_like(A)
    for i in range(n):
        idx = tuple([i] * k)
        D = D.at[idx].set(degrees[i])

    return D - A


# ===================================================================
# Tensor unfolding helpers (JIT-compatible)
# ===================================================================


@jax.jit
def _unfold_tensor(A_tensor: jax.Array) -> jax.Array:
    """Mode-1 unfolding of a tensor to a matrix.

    Given a tensor A of shape (n, n, ..., n) with *k* indices, produce the
    mode-1 unfolding of shape (n, n^{k-1}).  This is the standard
    matricisation used to linearise tensor dynamical systems.

    Parameters
    ----------
    A_tensor : jax.Array, shape (n, n, ..., n)
        Order-*k* tensor.

    Returns
    -------
    A_unf : jax.Array, shape (n, n^{k-1})
        Mode-1 unfolding.
    """
    n = A_tensor.shape[0]
    k = A_tensor.ndim
    return jnp.reshape(A_tensor, (n, n ** (k - 1)))


@jax.jit
def _unfold_to_square(A_tensor: jax.Array) -> jax.Array:
    """Unfold a tensor to a *square* matrix for control analysis.

    For higher-order tensors (k > 2) we linearise the multilinear dynamics
    ``x_dot_i = sum_{j,...} A[i,j,...] x_j x_k ...`` at a generic operating
    point ``x* = (1, 2, ..., n)``.  The Jacobian of the tensor contraction
    with respect to x evaluated at x* is:

        A_eff[i, j] = (k-1) * sum_{i3,...,ik} A[i, j, i3, ..., ik]
                       * x*[i3] * ... * x*[ik]

    This preserves the higher-order coupling information that a naive partial
    trace loses, making the resulting system generically controllable when the
    hypergraph structure supports it.

    For order-2 tensors this is a no-op (the tensor *is* the matrix).

    Parameters
    ----------
    A_tensor : jax.Array, shape (n, n, ..., n)

    Returns
    -------
    A_sq : jax.Array, shape (n, n)
    """
    k = A_tensor.ndim
    n = A_tensor.shape[0]
    if k == 2:
        return A_tensor

    # Generic operating point — distinct values avoid accidental symmetry.
    # Normalise by n to keep the spectral radius of A_eff moderate and
    # prevent numerical overflow in exp(A*T) during Gramian computation.
    x_star = jnp.arange(1, n + 1, dtype=A_tensor.dtype) / n

    # Contract trailing modes (2, 3, ..., k-1) with x_star sequentially.
    # Each contraction reduces the tensor order by 1.
    result = A_tensor
    for mode_idx in range(k - 1, 1, -1):
        # Contract the last remaining trailing mode with x_star.
        result = jnp.tensordot(result, x_star, axes=([mode_idx], [0]))

    # Multiply by (k-1) for the Jacobian prefactor (from the product rule:
    # there are k-1 positions where x_j can appear in the multilinear form,
    # and by tensor symmetry each contributes equally).
    return (k - 1) * result


# ===================================================================
# 3. Tensor Kalman rank condition
# ===================================================================


@partial(jax.jit, static_argnames=("num_terms",))
def tensor_kalman_rank(
    A_tensor: jax.Array,
    B: jax.Array,
    num_terms: Optional[int] = None,
) -> Tuple[jax.Array, jax.Array]:
    """Kalman rank condition for a tensor dynamical system.

    The tensor system  dx/dt = A \\otimes x + B u  is linearised via mode-1
    unfolding to  dx/dt = A_unf x + B u.  Controllability is then tested
    by the classical rank condition on the controllability matrix

        C = [B,  A_unf B,  A_unf^2 B,  ...,  A_unf^{n-1} B].

    Parameters
    ----------
    A_tensor : jax.Array, shape (n, n, ..., n)
        Order-*k* adjacency (or Laplacian) tensor.
    B : jax.Array, shape (n, m)
        Input matrix.
    num_terms : int, optional
        Number of terms in the controllability matrix (default: *n*).

    Returns
    -------
    rank : jax.Array, scalar int
        Numerical rank of the controllability matrix.
    is_controllable : jax.Array, scalar bool
        True iff rank == n.
    """
    A_sq = _unfold_to_square(A_tensor)
    n = A_sq.shape[0]
    if num_terms is None:
        num_terms = n

    return _kalman_rank_from_matrix(A_sq, B, num_terms)


@partial(jax.jit, static_argnames=("num_terms",))
def _kalman_rank_from_matrix(
    A: jax.Array,
    B: jax.Array,
    num_terms: int,
) -> Tuple[jax.Array, jax.Array]:
    """Compute the Kalman controllability rank from an (n, n) matrix A."""
    n = A.shape[0]
    m = B.shape[1]

    # Build controllability matrix [B, AB, A^2 B, ..., A^{k-1} B].
    def _scan_fn(carry, _):
        A_power_B = carry
        next_col = A @ A_power_B
        return next_col, A_power_B

    _, columns = lax.scan(_scan_fn, B, xs=None, length=num_terms)
    # columns has shape (num_terms, n, m) — stack along the column axis.
    C = jnp.reshape(jnp.transpose(columns, (1, 0, 2)), (n, num_terms * m))

    # Numerical rank via SVD.
    sv = jnp.linalg.svd(C, compute_uv=False)
    tol = sv[0] * n * jnp.finfo(C.dtype).eps
    rank = jnp.sum(sv > tol)
    is_controllable = rank >= n
    return rank, is_controllable


# ===================================================================
# 4. Minimum driver nodes
# ===================================================================


def _minimum_driver_nodes_impl(
    hg,
) -> Tuple[jax.Array, int]:
    """Find a minimum set of driver nodes for full controllability (internal).

    Uses the tensor-unfolding approach: unfold the adjacency tensor to an
    (n x n) matrix, then greedily add driver nodes until the Kalman rank
    reaches *n*.

    Returns
    -------
    driver_node_indices : jax.Array, shape (num_drivers,)
        Indices of driver nodes (sorted).
    num_driver_nodes : int
        Number of driver nodes.
    """
    A_tensor = adjacency_tensor(hg)
    A_sq = _unfold_to_square(A_tensor)
    n = A_sq.shape[0]

    # Greedy algorithm: start with no drivers, add the node that increases
    # the Kalman rank the most, repeat until full rank.
    driver_set: list[int] = []
    remaining = set(range(n))

    while True:
        if len(driver_set) == n:
            break

        best_node = -1
        best_rank = -1

        for candidate in sorted(remaining):
            trial_drivers = sorted(driver_set + [candidate])
            B_trial = jnp.eye(n, dtype=A_sq.dtype)[:, jnp.array(trial_drivers)]
            rank_val, is_ctrl = _kalman_rank_from_matrix(A_sq, B_trial, n)
            rank_int = int(rank_val)

            if rank_int > best_rank:
                best_rank = rank_int
                best_node = candidate

            if bool(is_ctrl):
                # This candidate achieves full rank — add it and stop.
                best_node = candidate
                break

        driver_set.append(best_node)
        remaining.discard(best_node)

        # Check if we have full rank now.
        B_check = jnp.eye(n, dtype=A_sq.dtype)[:, jnp.array(sorted(driver_set))]
        _, is_ctrl = _kalman_rank_from_matrix(A_sq, B_check, n)
        if bool(is_ctrl):
            break

    driver_indices = jnp.array(sorted(driver_set), dtype=jnp.int32)
    return driver_indices, len(driver_set)


def minimum_driver_nodes(
    hg,
) -> int:
    """Find the minimum number of driver nodes for full controllability.

    Uses the tensor-unfolding approach: unfold the adjacency tensor to an
    (n x n) matrix, then greedily add driver nodes until the Kalman rank
    reaches *n*.

    Parameters
    ----------
    hg : hgx.Hypergraph
        Hypergraph object.

    Returns
    -------
    num_driver_nodes : int
        Minimum number of driver nodes needed for controllability.
    """
    _require_hgx("minimum_driver_nodes")
    _, num_drivers = _minimum_driver_nodes_impl(hg)
    return num_drivers


# ===================================================================
# 5. Control energy
# ===================================================================


@partial(jax.jit, static_argnames=("num_steps",))
def _controllability_gramian(
    A: jax.Array,
    B: jax.Array,
    T: jax.Array,
    num_steps: int = 200,
) -> jax.Array:
    """Approximate the finite-horizon controllability Gramian.

    W(T) = int_0^T exp(A t) B B^T exp(A^T t) dt

    via the trapezoidal rule with *num_steps* points.

    Parameters
    ----------
    A : (n, n) array
    B : (n, m) array
    T : scalar
    num_steps : int

    Returns
    -------
    W : (n, n) array
        Controllability Gramian.
    """
    dt = T / num_steps
    ts = jnp.linspace(0.0, T, num_steps + 1)

    def _integrand(t):
        eAt = jax.scipy.linalg.expm(A * t)
        M = eAt @ B
        return M @ M.T

    # Trapezoidal rule via scan for JIT compatibility.
    def _scan_fn(W_acc, t):
        val = _integrand(t)
        return W_acc + val * dt, None

    # Endpoints get half weight (trapezoidal correction).
    W0 = _integrand(ts[0]) * (dt / 2.0)
    Wn = _integrand(ts[-1]) * (dt / 2.0)

    # Interior points.
    interior = ts[1:-1]

    def _scan_interior(W_acc, t):
        return W_acc + _integrand(t) * dt, None

    n = A.shape[0]
    W_interior, _ = lax.scan(_scan_interior, jnp.zeros((n, n), dtype=A.dtype), interior)

    return W0 + W_interior + Wn


def control_energy(
    hg,
    driver_nodes: jax.Array,
    xf: jax.Array,
    T: float,
    x0: Optional[jax.Array] = None,
) -> jax.Array:
    """Minimum control energy for steering hypergraph dynamics to *xf*.

    The system is linearised via tensor unfolding:  dx/dt = A x + B u.
    The minimum-energy control that steers x(0) = x0 to x(T) = xf has cost

        E* = (xf - exp(AT) x0)^T  W(T)^{-1}  (xf - exp(AT) x0)

    where W(T) is the finite-horizon controllability Gramian.

    Parameters
    ----------
    hg : hgx.Hypergraph
        Hypergraph object.
    driver_nodes : (m,) int array
        Indices of driver (input) nodes.
    xf : (n,) array
        Target state.
    T : float
        Time horizon (must be > 0).
    x0 : (n,) array, optional
        Initial state.  If *None*, defaults to the zero vector.

    Returns
    -------
    energy : jax.Array, scalar
        Minimum control energy.
    """
    _require_hgx("control_energy")

    A_tensor = adjacency_tensor(hg)
    A_sq = _unfold_to_square(A_tensor)
    n = A_sq.shape[0]

    driver_nodes = jnp.asarray(driver_nodes, dtype=jnp.int32)
    B = jnp.eye(n, dtype=A_sq.dtype)[:, driver_nodes]

    if x0 is None:
        x0 = jnp.zeros(n, dtype=A_sq.dtype)

    return _control_energy_impl(A_sq, B, x0, xf, jnp.float64(T))


@jax.jit
def _control_energy_impl(
    A: jax.Array,
    B: jax.Array,
    x0: jax.Array,
    xf: jax.Array,
    T: jax.Array,
) -> jax.Array:
    """JIT-compiled core of :func:`control_energy`."""
    eAT = jax.scipy.linalg.expm(A * T)
    delta = xf - eAT @ x0

    W = _controllability_gramian(A, B, T)

    # Regularise for numerical stability.
    n = A.shape[0]
    W_reg = W + 1e-10 * jnp.eye(n, dtype=W.dtype)

    # E* = delta^T W^{-1} delta
    W_inv_delta = jnp.linalg.solve(W_reg, delta)
    return delta @ W_inv_delta


# ===================================================================
# 6. Controllability profile
# ===================================================================


def controllability_profile(
    hg,
    driver_nodes: Optional[jax.Array] = None,
) -> Tuple[jax.Array, jax.Array]:
    """Per-node controllability metrics for a hypergraph.

    For each node *i*, compute:
    - The energy needed to reach a unit perturbation e_i from the origin
      (using the controllability Gramian with T = 1).
    - The Kalman rank when node *i* is the sole driver.

    Nodes that require high energy or yield low rank are "hard to control".

    Parameters
    ----------
    hg : hgx.Hypergraph
        Hypergraph object.
    driver_nodes : (m,) int array, optional
        If given, use these as the driver nodes.  Otherwise every node is
        tested individually as a sole driver.

    Returns
    -------
    energies : jax.Array, shape (n_nodes,)
        Minimum control energy to reach a unit perturbation at each node.
    ranks : jax.Array, shape (n_nodes,)
        Kalman rank when each node is the sole driver (or with the provided
        driver set).
    """
    _require_hgx("controllability_profile")

    A_tensor = adjacency_tensor(hg)
    A_sq = _unfold_to_square(A_tensor)
    n = A_sq.shape[0]

    if driver_nodes is not None:
        driver_nodes = jnp.asarray(driver_nodes, dtype=jnp.int32)
        B = jnp.eye(n, dtype=A_sq.dtype)[:, driver_nodes]
        rank_val, _ = _kalman_rank_from_matrix(A_sq, B, n)
        ranks = jnp.full(n, rank_val, dtype=jnp.int32)
        energies = _compute_energies_for_all_targets(A_sq, B, n)
    else:
        # Test each node individually as a sole driver.
        energies_list = []
        ranks_list = []
        for i in range(n):
            B_i = jnp.eye(n, dtype=A_sq.dtype)[:, i : i + 1]  # (n, 1)
            rank_i, _ = _kalman_rank_from_matrix(A_sq, B_i, n)
            ranks_list.append(rank_i)

            energy_i = _single_target_energy(A_sq, B_i, i, n)
            energies_list.append(energy_i)

        energies = jnp.array(energies_list)
        ranks = jnp.array(ranks_list, dtype=jnp.int32)

    return energies, ranks


@partial(jax.jit, static_argnames=("n",))
def _compute_energies_for_all_targets(
    A: jax.Array,
    B: jax.Array,
    n: int,
) -> jax.Array:
    """Compute energy to reach each unit vector e_i from the origin."""
    x0 = jnp.zeros(n, dtype=A.dtype)
    T = jnp.ones((), dtype=A.dtype)

    W = _controllability_gramian(A, B, T)
    W_reg = W + 1e-10 * jnp.eye(n, dtype=W.dtype)

    # For target e_i, energy = e_i^T W^{-1} e_i = W_inv[i, i].
    W_inv = jnp.linalg.inv(W_reg)
    return jnp.diag(W_inv)


@partial(jax.jit, static_argnames=("target_node", "n"))
def _single_target_energy(
    A: jax.Array,
    B: jax.Array,
    target_node: int,
    n: int,
) -> jax.Array:
    """Energy to reach unit perturbation at a single target node."""
    x0 = jnp.zeros(n, dtype=A.dtype)
    xf = jnp.zeros(n, dtype=A.dtype).at[target_node].set(1.0)
    T = jnp.ones((), dtype=A.dtype)
    return _control_energy_impl(A, B, x0, xf, T)


# ===================================================================
# 7. Hypergraph -> linear system conversion
# ===================================================================


def hypergraph_linear_system(
    hg,
    driver_nodes: Optional[jax.Array] = None,
) -> Tuple[jax.Array, jax.Array]:
    """Convert a hypergraph to a standard (A, B) linear system.

    The adjacency tensor is unfolded to a square (n x n) matrix A.
    The input matrix B is constructed from the driver nodes (one-hot columns).
    If no driver nodes are specified, the minimum driver node set is computed
    automatically.

    Parameters
    ----------
    hg : hgx.Hypergraph
        Hypergraph object.
    driver_nodes : (m,) int array, optional
        Indices of driver (input) nodes.  If *None*, the minimum driver set
        is computed via :func:`minimum_driver_nodes`.

    Returns
    -------
    A : jax.Array, shape (n, n)
        Unfolded adjacency matrix.
    B : jax.Array, shape (n, m)
        Input selection matrix.
    """
    _require_hgx("hypergraph_linear_system")

    A_tensor = adjacency_tensor(hg)
    A_sq = _unfold_to_square(A_tensor)
    n = A_sq.shape[0]

    if driver_nodes is None:
        driver_nodes, _ = _minimum_driver_nodes_impl(hg)

    driver_nodes = jnp.asarray(driver_nodes, dtype=jnp.int32)
    B = jnp.eye(n, dtype=A_sq.dtype)[:, driver_nodes]

    return A_sq, B


# ===================================================================
# HypergraphControlSystem module
# ===================================================================


class HypergraphControlSystem(eqx.Module):
    """A continuous-time linear dynamical system on a hypergraph.

    Wraps the unfolded adjacency matrix, input matrix, and driver node
    indices into an Equinox module for use with other ``jaxctrl`` routines
    and Diffrax integrators.

    The dynamics are::

        dx/dt = A x + B u(t)

    where A comes from the tensor unfolding of the hypergraph adjacency and
    B selects the driver (input) nodes.

    Attributes
    ----------
    adjacency : jax.Array, shape (n, n)
        Unfolded adjacency matrix.
    input_matrix : jax.Array, shape (n, m)
        Input selection matrix.
    driver_nodes : jax.Array, shape (m,)
        Indices of driver nodes.
    """

    adjacency: jax.Array
    input_matrix: jax.Array
    driver_nodes: jax.Array

    def __init__(
        self,
        hg=None,
        driver_nodes: Optional[jax.Array] = None,
        *,
        adjacency: Optional[jax.Array] = None,
        input_matrix: Optional[jax.Array] = None,
    ):
        """Create a hypergraph control system.

        Can be constructed either from an hgx Hypergraph or from explicit
        matrices.

        Parameters
        ----------
        hg : hgx.Hypergraph, optional
            Hypergraph object.  If given, ``adjacency`` and ``input_matrix``
            are computed automatically.
        driver_nodes : (m,) int array, optional
            Driver node indices.  If *hg* is given and this is None, the
            minimum driver set is computed automatically.
        adjacency : (n, n) array, optional
            Explicit unfolded adjacency matrix (use when *hg* is not given).
        input_matrix : (n, m) array, optional
            Explicit input matrix (use when *hg* is not given).
        """
        if hg is not None:
            _require_hgx("HypergraphControlSystem")
            A_sq, B = hypergraph_linear_system(hg, driver_nodes=driver_nodes)
            if driver_nodes is None:
                driver_nodes, _ = _minimum_driver_nodes_impl(hg)
            self.adjacency = A_sq
            self.input_matrix = B
            self.driver_nodes = jnp.asarray(driver_nodes, dtype=jnp.int32)
        elif adjacency is not None and input_matrix is not None:
            self.adjacency = jnp.asarray(adjacency)
            self.input_matrix = jnp.asarray(input_matrix)
            if driver_nodes is not None:
                self.driver_nodes = jnp.asarray(driver_nodes, dtype=jnp.int32)
            else:
                # Infer driver nodes from nonzero columns of B.
                self.driver_nodes = jnp.where(
                    jnp.any(jnp.asarray(input_matrix) != 0, axis=0),
                    size=jnp.asarray(input_matrix).shape[1],
                )[0].astype(jnp.int32)
        else:
            raise ValueError(
                "Provide either an hgx Hypergraph or both "
                "'adjacency' and 'input_matrix'."
            )

    @eqx.filter_jit
    def vector_field(self, x: jax.Array, u: jax.Array) -> jax.Array:
        """Evaluate the dynamics dx/dt = A x + B u.

        Parameters
        ----------
        x : (n,) array
            State vector.
        u : (m,) array
            Control input vector.

        Returns
        -------
        dx : (n,) array
            Time derivative of the state.
        """
        return self.adjacency @ x + self.input_matrix @ u

    @eqx.filter_jit
    def simulate(
        self,
        x0: jax.Array,
        u: jax.Array,
        T: float,
    ) -> jax.Array:
        """Simulate the system using matrix exponential discretisation.

        Uses a zero-order hold on the control input over *num_steps* equal
        intervals spanning [0, T].

        Parameters
        ----------
        x0 : (n,) array
            Initial state.
        u : (num_steps, m) array
            Control inputs at each time step.
        T : float
            Total simulation time.

        Returns
        -------
        trajectory : (num_steps + 1, n) array
            State trajectory including the initial condition.
        """
        num_steps = u.shape[0]
        dt = T / num_steps
        A = self.adjacency
        B = self.input_matrix

        eAdt = jax.scipy.linalg.expm(A * dt)
        n = A.shape[0]

        # Discretised input matrix: A^{-1} (e^{A dt} - I) B
        # Use series expansion for numerical stability near singular A.
        I_n = jnp.eye(n, dtype=A.dtype)
        # B_d = A^{-1}(eAdt - I) B; computed via solve for stability.
        B_d = jnp.linalg.solve(A + 1e-12 * I_n, (eAdt - I_n) @ B)

        def _step(x, u_k):
            x_next = eAdt @ x + B_d @ u_k
            return x_next, x_next

        _, trajectory = lax.scan(_step, x0, u)
        # Prepend initial condition.
        trajectory = jnp.concatenate([x0[None, :], trajectory], axis=0)
        return trajectory

    def controllability(
        self,
    ) -> Tuple[jax.Array, jax.Array]:
        """Test Kalman controllability of the system.

        Returns
        -------
        rank : jax.Array, scalar int
            Rank of the controllability matrix.
        is_controllable : jax.Array, scalar bool
            True iff the system is controllable.
        """
        n = self.adjacency.shape[0]
        return _kalman_rank_from_matrix(self.adjacency, self.input_matrix, n)

    @eqx.filter_jit
    def gramian(self, T: jax.Array) -> jax.Array:
        """Compute the finite-horizon controllability Gramian W(T).

        Parameters
        ----------
        T : scalar
            Time horizon.

        Returns
        -------
        W : (n, n) array
            Controllability Gramian.
        """
        return _controllability_gramian(self.adjacency, self.input_matrix, T)

    @eqx.filter_jit
    def lqr(
        self,
        Q: jax.Array,
        R: jax.Array,
    ) -> Tuple[jax.Array, jax.Array]:
        """Solve the infinite-horizon continuous-time LQR problem.

        Minimises  J = int_0^inf (x^T Q x + u^T R u) dt  subject to
        dx/dt = A x + B u.

        Parameters
        ----------
        Q : (n, n) array
            State cost matrix (PSD).
        R : (m, m) array
            Input cost matrix (PD).

        Returns
        -------
        K : (m, n) array
            Optimal feedback gain (u = -K x).
        P : (n, n) array
            Solution of the CARE.
        """
        from jaxctrl._riccati import lqr as _lqr

        return _lqr(self.adjacency, self.input_matrix, Q, R)
