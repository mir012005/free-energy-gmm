"""
gmm_networks.py
───────────────
Neural network architectures shared by MCD, CMCD and LED.

Both algorithms need a network that maps (q, time_step_index) → R^d.
  - MCD  : output = score approximation  s_θ(q, k) ≈ ∇ log p_k(q)
  - CMCD : output = escorting drift       u_θ(q, λ_k)
  - LED  : output = escorting drift       u_θ(q, λ_k)  (same architecture)

Architecture (identical to the thesis, Section 5.C):
  [q ∈ R^d, emb_k ∈ R^emb_dim]  →  2 × ResNet block  →  Dense  →  R^d

ResNet block:
  x  →  FanOut(2)  →  [Identity | Dense + Softplus]  →  FanInSum  →  x'
"""

from __future__ import annotations
import jax
import jax.numpy as jnp
from jax.example_libraries.stax import (
    Dense, serial, Softplus, FanInSum, FanOut, Identity, parallel,
)
from typing import Callable


def build_score_network(
    space_dim: int,
    n_steps: int,
    emb_dim: int = 20,
) -> tuple[Callable, Callable]:
    """
    Build a score / drift network.

    Parameters
    ----------
    space_dim : dimension of q
    n_steps   : number of integration steps  (sets embedding table size)
    emb_dim   : dimension of the learnable time embedding

    Returns
    -------
    init_fn   : (rng, _) → (output_shape, params)
    apply_fn  : (params, q, step_index) → R^{space_dim}

    params = {
        "nn"        : stax parameter pytree,
        "emb"       : (n_steps, emb_dim) learnable table,
        "scale"     : scalar multiplier  (initialised to 0 → warm start),
    }
    """
    input_dim = space_dim + emb_dim

    resnet_block = serial(
        FanOut(2),
        parallel(Identity, serial(Dense(input_dim), Softplus)),
        FanInSum,
    )
    init_nn, apply_nn = serial(resnet_block, resnet_block, Dense(space_dim))

    def init_fn(rng, _=None):
        params = {}
        _, params["nn"]  = init_nn(rng, (input_dim,))
        rng, _           = jax.random.split(rng)
        # small random init for embedding, zero for scale
        params["emb"]    = jax.random.normal(rng, (n_steps, emb_dim)) * 0.05
        params["scale"]  = jnp.array(1.0)
        return (space_dim,), params

    def apply_fn(params, q: jnp.ndarray, step_idx: int) -> jnp.ndarray:
        """
        q         : (space_dim,)
        step_idx  : integer in [0, n_steps)
        returns   : (space_dim,)
        """
        emb    = params["emb"][step_idx]             # (emb_dim,)
        inputs = jnp.concatenate([q, emb])           # (space_dim + emb_dim,)
        out    = apply_nn(params["nn"], inputs)       # (space_dim,)
        return out * params["scale"]

    return init_fn, apply_fn
