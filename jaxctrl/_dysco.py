# Copyright 2024 jaxctrl contributors. Apache-2.0 license.
"""DYSCO — governing equations from latent dynamics via contrastive learning.

A JAX/Equinox implementation of the core of DYSCO (Muratore & Mathis,
arXiv:2606.13260, 2026): jointly recover the latent trajectory AND its governing
dynamics from nonlinearly-observed time-series.  It fuses the two halves jaxctrl
already has the pieces for:

  - a contrastive encoder (``_contrastive``: CEBRA) that learns the latent, and
  - a **structured-basis latent flow** (``_sysid``: a SINDy library) that is the
    predictor — so the learned dynamics are symbolic: ż = Θ(z) · Ξ.

Training is JEPA-style: the predictor advances the latent one step and must match
the (stop-gradient) encoding of the next observation, with a contrastive term to
prevent collapse and an L1 penalty on Ξ for parsimony.  Per DYSCO, the latent
dynamical system is identified up to an affine gauge (eigenvalues of the linear
part are gauge-invariant), which is what the rotational-recovery test checks.
"""

from __future__ import annotations

from typing import List, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax

from jaxctrl._contrastive import ContrastiveEncoder
from jaxctrl._sysid import polynomial_library


class LatentFlow(eqx.Module):
    """Structured-basis latent dynamics ż = Θ(z)·Ξ (a SINDy flow), with a
    one-step Euler predictor z_{t+1} ≈ z_t + dt·ż."""

    Xi: jax.Array                         # (n_terms, latent_dim)
    degree: int = eqx.field(static=True)
    dt: float = eqx.field(static=True)

    def flow(self, Z: jax.Array) -> jax.Array:
        return polynomial_library(Z, self.degree) @ self.Xi

    def step(self, Z: jax.Array) -> jax.Array:
        return Z + self.dt * self.flow(Z)

    def linear_part(self) -> jax.Array:
        """Jacobian of the flow at the origin = the linear dynamics matrix A
        (rows 1..k of Ξ, transposed).  Its eigenvalues are gauge-invariant."""
        k = self.Xi.shape[1]
        return self.Xi[1:1 + k, :].T


class _DyscoModule(eqx.Module):
    encoder: ContrastiveEncoder
    flow: LatentFlow


def _dysco_loss(module, anchors, positives, temperature, lam_contrast, lam_sparse):
    Zt = jax.vmap(module.encoder)(anchors)       # (B, k)
    Ztp1 = jax.vmap(module.encoder)(positives)   # (B, k)

    # JEPA predictive: advance the latent and match the (stop-grad) next encoding.
    Zpred = module.flow.step(Zt)
    pred = jnp.mean(jnp.sum((Zpred - jax.lax.stop_gradient(Ztp1)) ** 2, axis=1))

    # Contrastive anti-collapse on normalized latents (InfoNCE, in-batch negatives).
    A = Zt / (jnp.linalg.norm(Zt, axis=1, keepdims=True) + 1e-8)
    P = Ztp1 / (jnp.linalg.norm(Ztp1, axis=1, keepdims=True) + 1e-8)
    logits = (A @ P.T) / temperature
    B = A.shape[0]
    contrast = -jnp.mean(jax.nn.log_softmax(logits, axis=1)[jnp.arange(B), jnp.arange(B)])

    sparse = jnp.sum(jnp.abs(module.flow.Xi))
    return pred + lam_contrast * contrast + lam_sparse * sparse


@eqx.filter_jit
def _step(module, opt_state, optimizer, anchors, positives, temp, lam_c, lam_s):
    loss, grads = eqx.filter_value_and_grad(_dysco_loss)(
        module, anchors, positives, temp, lam_c, lam_s
    )
    updates, opt_state = optimizer.update(grads, opt_state, module)
    module = eqx.apply_updates(module, updates)
    return module, opt_state, loss


class DYSCO:
    """Contrastive recovery of latent dynamics + governing equation.

    Parameters
    ----------
    latent_dim : dimensionality of the recovered latent.
    library_degree : polynomial library degree for the latent flow (1 = linear).
    dt : Euler step for the predictor (match the data sampling interval).
    width, depth : encoder MLP size.
    temperature, lam_contrast, lam_sparse : InfoNCE temperature and the weights of
        the contrastive and L1-sparsity terms.
    learning_rate, n_steps, batch_size : Adam schedule.
    """

    def __init__(self, latent_dim: int = 2, library_degree: int = 1, dt: float = 0.1,
                 width: int = 64, depth: int = 2, temperature: float = 0.1,
                 lam_contrast: float = 1.0, lam_sparse: float = 1e-3,
                 learning_rate: float = 1e-3, n_steps: int = 2000,
                 batch_size: int = 256, key: Optional[jax.Array] = None):
        self.latent_dim = latent_dim
        self.library_degree = library_degree
        self.dt = dt
        self.width = width
        self.depth = depth
        self.temperature = temperature
        self.lam_contrast = lam_contrast
        self.lam_sparse = lam_sparse
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.batch_size = batch_size
        self._key = key if key is not None else jr.PRNGKey(0)
        self._module: Optional[_DyscoModule] = None

    def fit(self, X: jax.Array) -> List[float]:
        X = jnp.asarray(X)
        T, D = X.shape
        key, k_enc, k_xi = jr.split(self._key, 3)
        encoder = ContrastiveEncoder(D, self.latent_dim, self.width, self.depth,
                                     key=k_enc, normalize=False)
        n_terms = polynomial_library(jnp.zeros((1, self.latent_dim)), self.library_degree).shape[1]
        flow = LatentFlow(
            Xi=1e-2 * jr.normal(k_xi, (n_terms, self.latent_dim)),
            degree=self.library_degree, dt=self.dt,
        )
        module = _DyscoModule(encoder, flow)
        optimizer = optax.adam(self.learning_rate)
        opt_state = optimizer.init(eqx.filter(module, eqx.is_array))
        B = min(self.batch_size, T - 1)

        history = []
        for _ in range(self.n_steps):
            key, kb = jr.split(key)
            idx = jr.randint(kb, (B,), 0, T - 1)
            module, opt_state, loss = _step(
                module, opt_state, optimizer, X[idx], X[idx + 1],
                self.temperature, self.lam_contrast, self.lam_sparse,
            )
            history.append(float(loss))
        self._module = module
        return history

    def transform(self, X: jax.Array) -> jax.Array:
        """Embed observations -> latent trajectory (T, latent_dim)."""
        return jax.vmap(self._module.encoder)(jnp.asarray(X))

    def coefficients(self) -> jax.Array:
        """Governing-equation coefficients Ξ (n_terms, latent_dim)."""
        return self._module.flow.Xi

    def linear_part(self) -> jax.Array:
        """Linear dynamics matrix A (gauge-invariant eigenvalues)."""
        return self._module.flow.linear_part()

    def simulate(self, z0: jax.Array, n: int) -> jax.Array:
        """Roll the learned flow forward from z0 for n steps."""
        def body(z, _):
            z = self._module.flow.step(z[None])[0]
            return z, z
        _, traj = jax.lax.scan(body, jnp.asarray(z0), None, length=n)
        return traj

    def __repr__(self) -> str:
        return (f"DYSCO(latent_dim={self.latent_dim}, "
                f"library_degree={self.library_degree}, n_steps={self.n_steps})")
