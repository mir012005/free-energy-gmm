"""
gmm_ais.py
==========
Annealed Importance Sampling (AIS) — the baseline algorithm.

PHYSICAL IDEA
-------------
AIS (also called "standard Jarzynski estimator") simulates a particle
that is gradually switched from state pi_0 to state pi_1 while measuring
the work done during the switch.

By Jarzynski's non-equilibrium work theorem (1997):

    E[exp(-W)]  =  Z_1 / Z_0  =  exp(-Delta_F)

where:
    W       = work accumulated along one non-equilibrium trajectory
    E[...]  = average over many independent trajectories
    Z_0, Z_1 = partition functions of pi_0 and pi_1
    Delta_F = free energy difference (what we want to estimate)

No learning is involved. The algorithm is purely a simulation.

THE ALGORITHM (step by step)
-----------------------------
For each of N independent trajectories:

    1. Draw initial position:  q_0 ~ pi_0  (sample from starting distribution)

    2. For k = 0, 1, ..., K-1:
        a. Accumulate work:
               W += (lambda_{k+1} - lambda_k) * [log gamma_0(q_k) - log gamma_1(q_k)]
           This is the discrete approximation of the Clausius work:
               dW = dV/d_lambda * d_lambda = [log gamma_0(q) - log gamma_1(q)] * d_lambda

        b. Propagate the particle one step using the Euler-Maruyama scheme
           at the new lambda value lambda_{k+1}:
               q_{k+1} = q_k - dt * grad V_{lambda_{k+1}}(q_k) + sqrt(2*dt) * noise

    3. Return the accumulated work W for this trajectory.

After N trajectories: Z1/Z0 ≈ mean(exp(-W_i))

WHY AIS IS THE BASELINE
-----------------------
AIS is guaranteed to give an UNBIASED estimator (up to discretisation error):
    E[exp(-W)] = Z1/Z0  exactly as dt -> 0

However, it has HIGH VARIANCE when the distributions pi_0 and pi_1 are very
different (far apart, different shapes). This is the main motivation for the
learning-based methods (MCD, CMCD, LED) which reduce variance by learning
optimal corrections.

IMPLEMENTATION DETAILS
-----------------------
This implementation uses JAX for speed:
- jax.lax.scan replaces a Python for-loop with a compiled loop
  This is much faster because it avoids Python overhead at every step
- The evaluation is batched via gmm_eval_utils.batched_estimate
  to avoid running out of memory for large N or K
"""

from __future__ import annotations
import jax
import jax.numpy as jnp
import numpy as np

from gmm_config import PipelineConfig
from gmm_jax_physics import make_jax_potential
from gmm_eval_utils import batched_estimate


def _sample_q0(cfg: PipelineConfig, d: int, key):
    """
    Sample the initial particle position q_0 from pi_0.

    This uses the ancestral sampling procedure:
        1. Draw a component index from the mixture weights
        2. Draw from the Gaussian of that component

    The key is a JAX random key. We split it to get fresh keys for each
    random operation (component selection and Gaussian sampling).

    Parameters
    ----------
    cfg : PipelineConfig — contains gmm0 (the starting distribution)
    d   : int — dimension of the space
    key : JAX random key

    Returns
    -------
    q0  : array of shape (d,) — initial position
    key : updated JAX random key (so the caller can continue using it)
    """
    # Split the key into three: one updated key + two for the two random ops
    key, k1, k2 = jax.random.split(key, 3)

    # Step 1: randomly select a GMM component according to the weights
    comp = jax.random.choice(k1, jnp.arange(cfg.gmm0.n_components),
                              p=jnp.array(cfg.gmm0.weights))

    # Step 2: sample from the Gaussian of that component
    means = jnp.array(cfg.gmm0.means)   # shape (K0, d)
    covs  = jnp.array(cfg.gmm0.covs)    # shape (K0, d, d)
    q0    = means[comp] + jax.random.multivariate_normal(
                k2, jnp.zeros(d), covs[comp])
    # means[comp] is the mean; we add a random Gaussian displacement
    # with covariance covs[comp]

    return q0, key


def run(cfg: PipelineConfig, eval_batch_size: int = 256) -> dict:
    """
    Run AIS and return the virtual works for N independent trajectories.

    WHAT THIS FUNCTION DOES:
        For each trajectory i in {1, ..., N}:
            - Simulate a Langevin particle under the interpolated potential
            - Accumulate the Jarzynski work W_i
        Return all W_i values.

    The free energy estimate is: Z1/Z0 ≈ mean(exp(-W_i))

    Parameters
    ----------
    cfg            : PipelineConfig
                     Contains all simulation parameters:
                     - cfg.gmm0, cfg.gmm1 : the two distributions
                     - cfg.n_samples      : number of trajectories N
                     - cfg.n_steps        : number of time steps K = T/dt
                     - cfg.dt             : time step dt
                     - cfg.schedule       : the lambda schedule [0, ..., 1]
                     - cfg.seed           : random seed

    eval_batch_size : int, default 256
                     Number of trajectories per mini-batch (memory management).
                     Reduce if you get Out Of Memory errors.

    Returns
    -------
    dict with keys:
        "works" : np.ndarray of shape (n_samples,)
                  The virtual works W_i. Use mean(exp(-W_i)) for Z1/Z0.
        "trajs" : None
                  Trajectories are not stored (would use too much memory).
        "name"  : "AIS"
                  Algorithm identifier for plots and tables.
    """

    d     = cfg.dim                    # dimension of the space
    K     = cfg.n_steps                # number of Langevin steps
    dt    = cfg.dt                     # time step
    sched = jnp.array(cfg.schedule)   # lambda schedule: [lambda_0, ..., lambda_K]

    # Build the JAX energy functions (compiled for speed)
    V, grad_V, dV_dlam, log_g0, log_g1 = make_jax_potential(cfg.gmm0, cfg.gmm1)

    def compute_work_single(seed_val):
        """
        Simulate ONE AIS trajectory and return the accumulated work W.

        This function is designed to be vectorised with jax.vmap (see
        batched_estimate). It takes an integer seed and returns a scalar.

        Parameters
        ----------
        seed_val : int — random seed for this trajectory

        Returns
        -------
        scalar — the total work W accumulated along this trajectory
        """

        # Initialise random key from the integer seed
        key = jax.random.PRNGKey(seed_val)

        # Sample starting position from pi_0
        q0, key = _sample_q0(cfg, d, key)

        def step(carry, k):
            """
            One step of the AIS simulation.

            This function is called K times by jax.lax.scan (the compiled loop).
            Each call:
                1. Accumulates the work contribution at step k
                2. Propagates the particle to step k+1

            Parameters
            ----------
            carry : tuple (q, w, key)
                q   : current particle position, shape (d,)
                w   : accumulated work so far, scalar
                key : current random key

            k : int — current step index (provided by jax.lax.scan)

            Returns
            -------
            (q_new, w_new, key_new) : updated carry
            None                    : no per-step output needed
            """
            q, w, key = carry

            # Current and next lambda values
            lam_k   = sched[k]       # lambda at step k
            lam_kp1 = sched[k + 1]  # lambda at step k+1
            dlam    = lam_kp1 - lam_k  # lambda increment (positive, = 1/K for linear schedule)

            # --- Work accumulation (Jarzynski formula, eq. 5.9 in thesis) ---
            # dW = dV/d_lambda * d_lambda = [log gamma_0(q) - log gamma_1(q)] * dlam
            # This is the discrete Clausius work for switching lambda by dlam
            # while the particle stays at position q (before it moves).
            w += dlam * dV_dlam(q, lam_k)

            # --- Euler-Maruyama step at lambda_{k+1} ---
            # The particle moves under the NEW potential V_{lambda_{k+1}}.
            # This is the "one-step lag" that makes AIS non-equilibrium.
            key, sk = jax.random.split(key)   # fresh key for the noise
            noise   = jax.random.normal(sk, (d,))   # Gaussian noise G ~ N(0, I)
            gq      = grad_V(q, lam_kp1)            # force at new lambda

            # Discretised Langevin step:
            #   q_{k+1} = q_k  -  dt * grad V_{lambda_{k+1}}(q_k)  +  sqrt(2*dt) * G
            q_new = q - dt * gq + jnp.sqrt(2.0 * dt) * noise

            return (q_new, w, key), None  # return updated carry, no per-step output

        # jax.lax.scan runs the step function K times.
        # It is the JAX equivalent of:
        #   for k in range(K):
        #       carry = step(carry, k)
        # but compiled into a single fast kernel (no Python loop overhead).
        # jnp.arange(K) provides k = 0, 1, 2, ..., K-1 as the second argument.
        (_, w_K, _), _ = jax.lax.scan(step, (q0, 0.0, key), jnp.arange(K))
        # w_K is the total accumulated work after K steps

        return w_K

    # Run all N trajectories using the batched evaluation loop.
    # This handles memory management (mini-batches) and the progress bar.
    works = batched_estimate(
        compute_work_single,
        n_samples=cfg.n_samples,
        seed=cfg.seed,
        eval_batch_size=eval_batch_size,
        desc="AIS eval",
    )

    return {"works": works, "trajs": None, "name": "AIS"}
