"""
gmm_eval_utils.py
─────────────────
Shared batched evaluation loop used by MCD, CMCD and LED.

jax.vmap over n_samples trajectories allocates everything at once.
For n_samples=10000 and K=10000 this causes OOM or extreme slowness.

Solution: process trajectories in small batches (eval_batch_size),
accumulate works in a Python list, concatenate at the end.
This keeps memory bounded at eval_batch_size × K × d × 4 bytes.
"""

from __future__ import annotations
import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm


def batched_estimate(
    compute_work_single,   # (seed: int) -> W scalar  (jitted + vmapped internally)
    n_samples: int,
    seed: int,
    eval_batch_size: int = 256,
    desc: str = "Evaluating",
) -> np.ndarray:
    """
    Evaluate n_samples works in mini-batches of eval_batch_size.

    compute_work_single must be a JAX function: seed -> scalar work W.
    It will be vmapped over eval_batch_size seeds per batch.

    Returns works: np.ndarray of shape (n_samples,).
    """
    # Build a jitted vmap of a single-trajectory function
    compute_batch = jax.jit(jax.vmap(compute_work_single))

    key  = jax.random.PRNGKey(seed)
    all_works = []
    n_done    = 0

    pbar = tqdm(total=n_samples, desc=desc, leave=False)
    while n_done < n_samples:
        batch = min(eval_batch_size, n_samples - n_done)
        key, sk = jax.random.split(key)
        seeds   = jax.random.randint(sk, (batch,), 1, int(1e6))
        works_b = compute_batch(seeds)
        all_works.append(np.array(works_b))
        n_done += batch
        pbar.update(batch)
    pbar.close()

    return np.concatenate(all_works, axis=0)
