"""
gmm_networks.py
===============
The neural network architecture shared by MCD, CMCD and LED.

WHAT THE NETWORK DOES
---------------------
All three learning algorithms need a function that takes:
    - a particle position  q  in R^d  (where d is the dimension)
    - a time step index    k  in {0, 1, ..., K-1}

and returns a vector in R^d. Depending on the algorithm:
    - MCD  uses it as a score approximation:  s_theta(q, k) ≈ grad_q log p_k(q)
    - CMCD uses it as an escorting drift:      u_theta(q, lambda_k)
    - LED  uses it as an escorting drift:      u_theta(q, lambda_k)

In all cases, the output is a d-dimensional vector that modifies the
Langevin dynamics to reduce the variance of the free energy estimator.

WHY A NEURAL NETWORK?
---------------------
The optimal correction (score or drift) is unknown analytically for
arbitrary GMMs. A neural network can approximate any smooth function
given enough training data, so we use it as a universal function approximator.

The network is trained by minimising a loss function that measures how
far the current approximation is from the optimal one.

ARCHITECTURE
------------
The architecture follows Section 5.C of the thesis:

    Input: [q (d values), time embedding (emb_dim values)]
               |
    ResNet block 1
               |
    ResNet block 2
               |
    Dense layer  (d values)
               |
    Output: vector in R^d

Each ResNet block has the structure:
    x  ->  split into two copies
        copy 1: identity (unchanged)
        copy 2: Dense layer + Softplus activation
    add both copies back together  ->  output

The "residual" (skip) connection from copy 1 helps gradients flow during
training and allows the network to start near zero output (good initialisation
for physics problems where the correction should start small).

TIME EMBEDDING
--------------
The time step k is not fed directly as a number. Instead, each step k has a
dedicated learnable embedding vector of size emb_dim. This is analogous to
"positional encoding" in transformers.

The embedding table is an (n_steps, emb_dim) matrix where row k is the
embedding for step k. The network learns these embeddings during training.

Concretely: to evaluate the network at (q, k), we look up row k from the
embedding table and concatenate it with q before feeding to the ResNet.

WARM START
----------
The scale parameter is initialised to 0. This means the network outputs
zero at the start of training. This is intentional: without any learned
correction, the dynamics reduces to standard Langevin. Training then
gradually learns a useful correction, rather than starting from noise.
"""

from __future__ import annotations
import jax
import jax.numpy as jnp
from jax.example_libraries.stax import (
    Dense,      # standard fully-connected layer: output = W*x + b
    serial,     # chain layers in sequence: output = f_n(...f_2(f_1(x))...)
    Softplus,   # smooth activation: Softplus(x) = log(1 + exp(x))
    FanInSum,   # add two inputs element-wise: (a, b) -> a + b
    FanOut,     # duplicate input: x -> (x, x)
    Identity,   # pass-through: x -> x
    parallel,   # apply two functions to two inputs: (f, g)(a, b) = (f(a), g(b))
)
from typing import Callable


def build_score_network(
    space_dim: int,
    n_steps:   int,
    emb_dim:   int = 20,
) -> tuple[Callable, Callable]:
    """
    Build the score/drift neural network.

    Parameters
    ----------
    space_dim : int
        Dimension d of the particle position space.
        For a 1D problem d=1, for a 2D problem d=2, etc.

    n_steps : int
        Number of time steps K in the simulation.
        This determines the size of the time embedding table: (K, emb_dim).
        Each step k in {0, ..., K-1} has its own embedding vector.

    emb_dim : int, default 20
        Dimension of the learnable time embedding vectors.
        Larger values give the network more capacity to distinguish
        between different time steps, but increase memory usage.

    Returns
    -------
    init_fn : callable
        Function that initialises the network parameters.
        Signature: (rng_key, _) -> (output_shape, params)
        Call once before training: _, params = init_fn(rng_key)

    apply_fn : callable
        Function that evaluates the network.
        Signature: (params, q, step_idx) -> array of shape (d,)
        Call at every time step during training and evaluation.

    About the returned params dict
    -------------------------------
    params is a Python dict with three entries:
        params["nn"]    : the weights of the ResNet layers
                          (a nested dict of arrays managed by stax)
        params["emb"]   : array of shape (n_steps, emb_dim)
                          the learnable time embedding table
        params["scale"] : scalar multiplier, initialised to 0
                          allows warm-starting near zero output

    These are just arrays — no special objects needed. They can be saved
    with pickle, copied, averaged, etc.
    """

    # The network input is the concatenation of q (d values) and the
    # time embedding (emb_dim values)
    input_dim = space_dim + emb_dim

    # Build the ResNet block: x -> x + Dense(Softplus(Dense(x)))
    # Decomposed step by step:
    #   FanOut(2)             : duplicate x to get (x, x)
    #   parallel(Identity,    : leave the first copy unchanged
    #            Dense+Softplus): transform the second copy
    #   FanInSum              : add both copies back together
    #
    # This is the "pre-activation ResNet" variant.
    resnet_block = serial(
        FanOut(2),
        parallel(
            Identity,                              # skip connection: x unchanged
            serial(Dense(input_dim), Softplus),    # transformation branch
        ),
        FanInSum,    # residual addition: x_skip + x_transformed
    )

    # Full network: two ResNet blocks followed by a linear output projection
    # The stax library uses (init_fn, apply_fn) pairs for each layer
    init_nn, apply_nn = serial(
        resnet_block,          # first residual block
        resnet_block,          # second residual block
        Dense(space_dim),      # project back to d dimensions (linear, no activation)
    )

    def init_fn(rng, _=None):
        """
        Initialise all network parameters.

        Called once before training. Returns a params dict containing
        all learnable parameters initialised to their starting values.

        Parameters
        ----------
        rng : JAX random key — for reproducibility, always pass the same key
              to get the same initialisation.
        _   : ignored (kept for API compatibility with stax)

        Returns
        -------
        (output_shape, params) where:
            output_shape = (space_dim,)  — shape of the network output
            params       = dict with keys "nn", "emb", "scale"
        """
        params = {}

        # Initialise the ResNet weights using stax's default Glorot initialisation
        _, params["nn"] = init_nn(rng, (input_dim,))

        # Split the random key to get a fresh key for the embedding
        rng, _ = jax.random.split(rng)

        # Initialise the time embedding table with small random values
        # Small values (0.05 std) keep the initial network output near zero
        params["emb"] = jax.random.normal(rng, (n_steps, emb_dim)) * 0.05

        # Scale starts at 0 -> network output = 0 at initialisation
        # This is the "warm start": training begins with no correction,
        # which is the same as plain Langevin dynamics.
        params["scale"] = jnp.array(0.0)

        return (space_dim,), params

    def apply_fn(params, q: jnp.ndarray, step_idx: int) -> jnp.ndarray:
        """
        Evaluate the network at position q and time step step_idx.

        Steps:
            1. Look up the time embedding vector for step_idx
            2. Concatenate [q, embedding] to form the input vector
            3. Pass through the two ResNet blocks and the Dense layer
            4. Multiply by the scale parameter

        Parameters
        ----------
        params   : dict returned by init_fn (and updated by the optimiser)
        q        : array of shape (d,) — current particle position
        step_idx : int in [0, n_steps) — current time step index

        Returns
        -------
        array of shape (d,) — the score or drift correction vector

        Note on step_idx
        ----------------
        Inside jax.lax.scan, step_idx is a JAX integer (not a Python int).
        Indexing params["emb"][step_idx] works because JAX supports dynamic
        integer indexing on arrays.
        """
        # Step 1: look up the time embedding for this step
        emb = params["emb"][step_idx]                    # shape (emb_dim,)

        # Step 2: build the input vector [q, emb]
        inputs = jnp.concatenate([q, emb])               # shape (d + emb_dim,)

        # Step 3: pass through the ResNet
        out = apply_nn(params["nn"], inputs)              # shape (d,)

        # Step 4: multiply by the learnable scale (starts at 0)
        return out * params["scale"]

    return init_fn, apply_fn
