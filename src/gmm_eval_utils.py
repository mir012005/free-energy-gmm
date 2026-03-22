"""
gmm_eval_utils.py
=================
Shared batched evaluation loop used by all four algorithms.

THE MEMORY PROBLEM
------------------
The naive way to evaluate N=10,000 trajectories of K=1,000 steps each
in a d=2 dimensional space would be:

    jax.vmap(compute_work)(all_seeds)   # vectorise over all N trajectories

This would try to allocate N * K * d = 10,000 * 1,000 * 2 = 20 million
floating point numbers simultaneously in GPU/CPU memory.
At 4 bytes per number, that is 80 MB for the positions alone.
With gradients and intermediate values, the total can easily exceed
available memory (Out Of Memory error, OOM).

THE SOLUTION: MINI-BATCHES
---------------------------
Instead of processing all N trajectories at once, we process them in small
groups of size eval_batch_size (default: 256).

For each mini-batch:
    1. Generate a batch of 256 random seeds
    2. vmap compute_work over these 256 seeds (vectorised, fast)
    3. Copy the resulting 256 work values to regular NumPy (CPU memory)
    4. Free the GPU memory

After all batches, concatenate the results.

Memory usage is now bounded by:
    eval_batch_size * K * d * 4 bytes = 256 * 1000 * 2 * 4 = 2 MB

instead of 80 MB+ for the full batch. This trades a small overhead (multiple
kernel launches) for a large reduction in memory usage.

RANDOM SEEDS IN JAX
-------------------
JAX uses explicit random keys for reproducibility. Unlike NumPy where you
set a global random seed once, in JAX every random operation requires a key.

To generate N independent trajectories, we:
    1. Start from a master key derived from cfg.seed
    2. Split it to get a new key + a "subkey" for generating seeds
    3. Generate eval_batch_size random integer seeds from the subkey
    4. Each seed is used to initialise one trajectory's key

This ensures perfect reproducibility: the same cfg.seed always gives
the same trajectories.
"""

from __future__ import annotations
import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm


def batched_estimate(
    compute_work_single,    # a JAX function: seed (int) -> W (scalar)
    n_samples:       int,
    seed:            int,
    eval_batch_size: int = 256,
    desc:            str = "Evaluating",
) -> np.ndarray:
    """
    Evaluate the virtual work W for n_samples independent trajectories,
    processing them in mini-batches to avoid running out of memory.

    WHAT THIS FUNCTION DOES:
    For each trajectory i in {1, ..., n_samples}:
        - Start the Langevin particle at a random position drawn from pi_0
        - Simulate the trajectory under the interpolated potential V_lambda
        - Accumulate the work W_i along the way
        - Return all W_i values as a NumPy array

    The estimator of Z1/Z0 is then:  mean(exp(-W_i))

    USAGE:
    This function is called inside estimate() in each algorithm file.
    You pass it a function compute_work_single(seed) -> scalar W,
    and it handles all the batching, JAX compilation, and progress display.

    Parameters
    ----------
    compute_work_single : callable
        A JAX function that takes an integer seed and returns a scalar
        virtual work W for one trajectory.

        The seed is used to initialise the random number generator for
        that trajectory — different seeds give independent trajectories.

        Important: this function must be JAX-compatible (no Python side-effects,
        no dynamic Python control flow). It is internally JIT-compiled and
        vectorised with vmap.

    n_samples : int
        Total number of trajectories to simulate.
        More trajectories = better statistical estimate of Z1/Z0.
        Typical values: 2,000 (fast) to 10,000 (accurate).

    seed : int
        Master random seed for reproducibility.
        Using the same seed always gives the same set of trajectories.

    eval_batch_size : int, default 256
        Number of trajectories per mini-batch.
        Larger values are faster (more parallelism) but use more memory.
        Reduce to 64 or 32 if you get Out Of Memory errors.
        Increase to 512 or 1024 if you have plenty of GPU memory.

    desc : str
        Label for the progress bar (e.g. "AIS eval", "MCD eval").

    Returns
    -------
    np.ndarray of shape (n_samples,)
        The virtual works W_1, W_2, ..., W_{n_samples}.
        From these, compute:
            ratio_estimate = np.mean(np.exp(-works))   ->  Z1/Z0
            dF_estimate    = -np.log(ratio_estimate)   ->  Delta_F

    Example
    -------
        def compute_work_single(seed):
            # ... simulate one trajectory ...
            return W   # scalar JAX value

        works = batched_estimate(
            compute_work_single,
            n_samples=10_000,
            seed=42,
            eval_batch_size=256,
            desc="AIS eval",
        )
        Z1_over_Z0 = np.mean(np.exp(-works))
    """

    # jax.vmap(fn)(seeds) applies fn to each element of seeds independently
    # and in parallel. Here we vectorise compute_work_single over a batch
    # of integer seeds.
    # jax.jit then compiles the vectorised function for speed.
    # The first call will trigger compilation (slow). Subsequent calls reuse
    # the compiled version (fast).
    compute_batch = jax.jit(jax.vmap(compute_work_single))

    # Master random key — all trajectory seeds are derived from this
    key      = jax.random.PRNGKey(seed)
    all_works = []   # accumulate results from each mini-batch
    n_done   = 0     # how many trajectories we have computed so far

    # tqdm displays a progress bar in the terminal
    pbar = tqdm(total=n_samples, desc=desc, leave=False)

    while n_done < n_samples:
        # How many trajectories in this batch?
        # (the last batch might be smaller than eval_batch_size)
        batch = min(eval_batch_size, n_samples - n_done)

        # Split the master key: key is updated, sk is used for this batch
        # This is JAX's way of generating random numbers without a global state
        key, sk = jax.random.split(key)

        # Generate `batch` random integer seeds in [1, 1e6)
        # Each seed will initialise one independent trajectory
        seeds = jax.random.randint(sk, (batch,), 1, int(1e6))

        # Run the batch: vmap evaluates all `batch` trajectories in parallel
        works_b = compute_batch(seeds)   # shape (batch,) JAX array on GPU/CPU

        # Move the results from JAX (possibly GPU) to regular NumPy (CPU)
        # This also frees the GPU memory used by this batch
        all_works.append(np.array(works_b))

        n_done += batch
        pbar.update(batch)

    pbar.close()

    # Concatenate all batches into a single array of shape (n_samples,)
    return np.concatenate(all_works, axis=0)
