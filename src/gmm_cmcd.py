"""
gmm_cmcd.py
===========
Controlled Monte Carlo Diffusion (CMCD) — Section 5.3.3 of the thesis.

DIFFERENCE FROM MCD
--------------------
MCD learns the BACKWARD kernel while keeping the FORWARD dynamics fixed
(standard Langevin). The trajectories always follow standard EM, and only
the reweighting (importance weights) is improved by learning.

CMCD instead learns a CONTROLLED FORWARD dynamics. The forward kernel is
modified by a learned drift u_theta:

    q_{k+1} = q_k  -  dt * grad V_{lambda_{k+1}}(q_k)
              +  dt * Lambda'(s) * u_theta(q_k, k)      <- new drift term
              +  sqrt(2*dt) * G_k

where Lambda'(s) = dLambda/ds = 1/T (for linear scheduling).

The backward kernel is then the time-reversal of this new controlled process:

    B^theta_k(q | q') = N(q' - dt * grad V_{lambda_k}(q')
                            + dt * Lambda' * u_theta(q', max(k-1, 0)), 2*dt*I)

KEY ADVANTAGE over MCD: the backward kernel does NOT require a score network.
It only uses the same drift network u_theta. This makes CMCD "score-free"
while still being able to learn a better importance weight.

WHY THIS WORKS
--------------
By Girsanov's theorem, modifying the forward dynamics changes the path measure.
The importance weight formula (eq. 5.29) accounts for this change:

    IW = gamma_1(q_K) / gamma_0(q_0)
         * product_k B^theta_k(q_k | q_{k+1}) / F^theta_{k+1}(q_{k+1} | q_k)

At the optimal drift u^*, the controlled forward process exactly samples
from the "bridge" distribution connecting pi_0 and pi_1 — all trajectories
have the same work W = Delta_F and the variance is zero.

TRAINING
---------
Same ELBO objective as MCD, but now both the forward AND backward kernels
depend on theta (through u_theta). The ELBO is:

    ELBO(theta) = E_Q^theta[ log gamma_1(q_K) - log gamma_0(q_0)
                             + sum_k [log B^theta_k - log F^theta_{k+1}] ]

This is harder to optimise than MCD because the trajectory distribution Q^theta
also depends on theta. New trajectories must be generated at each epoch.

GRADIENT CLIPPING
-----------------
Differentiating through K steps of stochastic dynamics can cause gradient
explosion. We use gradient clipping (optax.clip_by_global_norm) to cap the
gradient norm, preventing unstable updates.

SMALL K DURING TRAINING
------------------------
Like MCD, we use K=64 steps during training to avoid OOM. Unlike MCD, there
is no embedding index mapping at evaluation time — CMCD always evaluates with
its own K_cmcd and dt_cmcd, regardless of the evaluation grid.
"""

from __future__ import annotations
import jax
import jax.numpy as jnp
import numpy as np
import optax
from tqdm import tqdm

from gmm_config import PipelineConfig
from gmm_jax_physics import make_jax_potential
from gmm_networks import build_score_network
from gmm_eval_utils import batched_estimate


def _sample_q0(cfg, d, key):
    """Sample initial position from pi_0. See gmm_ais.py for details."""
    key, k1, k2 = jax.random.split(key, 3)
    comp  = jax.random.choice(k1, jnp.arange(cfg.gmm0.n_components),
                               p=jnp.array(cfg.gmm0.weights))
    means = jnp.array(cfg.gmm0.means)
    covs  = jnp.array(cfg.gmm0.covs)
    q0    = means[comp] + jax.random.multivariate_normal(k2, jnp.zeros(d), covs[comp])
    return q0, key


def train(
    cfg:                  PipelineConfig,
    n_epochs:             int   = 5000,
    early_stop_patience:  int   = 500,
    batch_size:           int   = 128,
    lr_init:              float = 2e-4,
    lr_peak:              float = 1e-3,
    lr_end:               float = 1e-5,
    emb_dim:              int   = 20,
    grad_clip:            float = 0.5,
    K_cmcd:               int   = 64,
) -> tuple[dict, list[float], dict]:
    """
    Train the CMCD drift network u_theta.

    Parameters
    ----------
    cfg : PipelineConfig
    n_epochs : int — maximum training epochs
    early_stop_patience : int — stop if no improvement for this many epochs
    batch_size : int — trajectories per gradient step
    lr_init, lr_peak, lr_end : float — learning rate schedule
    grad_clip : float — maximum gradient norm (prevents explosion)
                        Lower values = more stable but slower training.
                        0.5 works well for 2D problems.
    K_cmcd : int — number of steps (kept small to avoid OOM)
    emb_dim : int — time embedding dimension

    Returns
    -------
    params : dict — trained network parameters
    loss_history : list — loss at each epoch
    meta : dict — {"K": K_cmcd, "dt": dt_cmcd}
    """
    d       = cfg.dim
    T       = cfg.T
    Lp      = 1.0 / T          # Lambda'(s) = 1/T for linear scheduling
    dt_cmcd = T / K_cmcd
    sched   = jnp.linspace(0.0, 1.0, K_cmcd + 1)
    meta    = {"K": K_cmcd, "dt": dt_cmcd}

    V, grad_V, _, log_g0, log_g1 = make_jax_potential(cfg.gmm0, cfg.gmm1)
    init_fn, apply_fn = build_score_network(d, K_cmcd, emb_dim)

    key = jax.random.PRNGKey(cfg.seed)
    key, sk = jax.random.split(key)
    _, params = init_fn(sk)

    def elbo_single(q0, params, key):
        """
        Compute the ELBO for one CONTROLLED trajectory.

        The drift u_theta modifies both the forward kernel F^theta AND
        the backward kernel B^theta, so both depend on the network.

        FORWARD KERNEL (eq. 5.10):
            F^theta_{k+1}(q_{k+1} | q_k) = N(q_k - dt * grad V_{lambda_{k+1}}(q_k)
                                               + dt * Lp * u_theta(q_k, k), 2*dt*I)

        BACKWARD KERNEL (eq. 5.28):
            B^theta_k(q_k | q_{k+1}) = N(q_{k+1} - dt * grad V_{lambda_k}(q_{k+1})
                                          + dt * Lp * u_theta(q_{k+1}, max(k-1,0)), 2*dt*I)

        Note: the backward kernel at step k uses lambda_k (NOT lambda_{k+1}).
        The backward time index is max(k-1, 0) to handle the boundary k=0.

        Returns
        -------
        scalar — ELBO for this trajectory (should approach -Delta_F at optimum)
        """
        def step(carry, k):
            q, acc, key = carry
            lam_k   = sched[k]        # lambda for backward kernel
            lam_kp1 = sched[k + 1]   # lambda for forward kernel

            # Evaluate drift network at current position
            u_q = apply_fn(params, q, k)   # shape (d,)

            # Controlled EM step (forward kernel with drift)
            key, sk = jax.random.split(key)
            noise   = jax.random.normal(sk, (d,))
            gq      = grad_V(q, lam_kp1)
            q_new   = (q
                       - dt_cmcd * gq                     # Langevin drift
                       + dt_cmcd * Lp * u_q               # learned drift
                       + jnp.sqrt(2.0 * dt_cmcd) * noise) # thermal noise

            # log F^theta_{k+1}(q_new | q)
            diff_f = q_new - q + dt_cmcd * gq - dt_cmcd * Lp * u_q
            log_f  = -(diff_f ** 2).sum() / (4.0 * dt_cmcd)

            # log B^theta_k(q | q_new)
            # Backward uses lambda_k and time index max(k-1, 0)
            bk     = jnp.maximum(k - 1, 0)                # backward time index
            u_qnew = apply_fn(params, q_new, bk)
            mean_b = (q_new
                      - dt_cmcd * grad_V(q_new, lam_k)    # grad at lambda_k (backward!)
                      + dt_cmcd * Lp * u_qnew)
            diff_b = q - mean_b
            log_b  = -(diff_b ** 2).sum() / (4.0 * dt_cmcd)

            acc += log_b - log_f
            return (q_new, acc, key), None

        (q_K, acc_K, _), _ = jax.lax.scan(step, (q0, 0.0, key), jnp.arange(K_cmcd))
        return log_g1(q_K) - log_g0(q0) + acc_K

    def loss_fn(seeds, params):
        def one(seed):
            k = jax.random.PRNGKey(seed)
            q0, k = _sample_q0(cfg, d, k)
            return elbo_single(q0, params, k)
        elbos = jax.vmap(one)(seeds)
        loss  = -elbos.mean()
        return loss, (loss,)

    lr_sched = optax.warmup_cosine_decay_schedule(
        init_value=lr_init, peak_value=lr_peak,
        warmup_steps=int(n_epochs * 0.2),
        decay_steps=n_epochs, end_value=lr_end,
    )
    # optax.chain applies multiple gradient transformations in sequence:
    # 1. Clip gradients to prevent explosion (critical for CMCD stability)
    # 2. Apply Adam optimiser
    optimizer = optax.chain(
        optax.clip_by_global_norm(grad_clip),   # clip gradient norm to grad_clip
        optax.adam(lr_sched),
    )
    opt_state = optimizer.init(params)
    grad_fn   = jax.jit(jax.grad(loss_fn, argnums=1, has_aux=True))

    loss_history, best_loss, patience_ctr = [], float("inf"), 0
    key = jax.random.PRNGKey(cfg.seed + 1)

    pbar = tqdm(range(n_epochs), desc="CMCD training")
    for epoch in pbar:
        key, sk = jax.random.split(key)
        seeds   = jax.random.randint(sk, (batch_size,), 1, int(1e6))
        grads, (loss,) = grad_fn(seeds, params)
        loss_val = float(loss)
        loss_history.append(loss_val)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        if loss_val < best_loss:
            best_loss, patience_ctr = loss_val, 0
        else:
            patience_ctr += 1
        if patience_ctr >= early_stop_patience:
            pbar.set_description(f"CMCD [early stop @ {epoch+1}]")
            break

    return params, loss_history, meta


def estimate(
    cfg:            PipelineConfig,
    params:         dict,
    meta:           dict,
    emb_dim:        int = 20,
    eval_batch_size: int = 256,
) -> dict:
    """
    Estimate Z1/Z0 using the trained CMCD drift network.

    Unlike MCD, CMCD always uses the same K_cmcd and dt_cmcd as training —
    there is no embedding index mapping. This is because the drift network
    was designed for a specific grid of K_cmcd steps.

    The work formula:
        W = log gamma_0(q_0) - log gamma_1(q_K)
            - sum_k [log B^theta_k(q_k | q_{k+1}) - log F^theta_{k+1}(q_{k+1} | q_k)]

    Parameters
    ----------
    cfg    : PipelineConfig — evaluation configuration
    params : dict — trained network parameters from train()
    meta   : dict — {"K": K_cmcd, "dt": dt_cmcd} from train()
    emb_dim : int
    eval_batch_size : int

    Returns
    -------
    dict with keys "works", "trajs" (None), "name" ("CMCD")
    """
    d       = cfg.dim
    T       = cfg.T
    Lp      = 1.0 / T
    K_cmcd  = meta["K"]
    dt_cmcd = meta["dt"]
    sched   = jnp.linspace(0.0, 1.0, K_cmcd + 1)

    V, grad_V, _, log_g0, log_g1 = make_jax_potential(cfg.gmm0, cfg.gmm1)
    _, apply_fn = build_score_network(d, K_cmcd, emb_dim)

    def compute_work_single(seed_val):
        key = jax.random.PRNGKey(seed_val)
        q0, key = _sample_q0(cfg, d, key)

        def step(carry, k):
            q, acc, key = carry
            lam_k   = sched[k]
            lam_kp1 = sched[k + 1]

            u_q     = apply_fn(params, q, k)
            key, sk = jax.random.split(key)
            noise   = jax.random.normal(sk, (d,))
            gq      = grad_V(q, lam_kp1)
            q_new   = (q - dt_cmcd * gq
                         + dt_cmcd * Lp * u_q
                         + jnp.sqrt(2.0 * dt_cmcd) * noise)

            diff_f  = q_new - q + dt_cmcd * gq - dt_cmcd * Lp * u_q
            log_f   = -(diff_f ** 2).sum() / (4.0 * dt_cmcd)

            bk      = jnp.maximum(k - 1, 0)
            u_qnew  = apply_fn(params, q_new, bk)
            mean_b  = (q_new - dt_cmcd * grad_V(q_new, lam_k)
                              + dt_cmcd * Lp * u_qnew)
            diff_b  = q - mean_b
            log_b   = -(diff_b ** 2).sum() / (4.0 * dt_cmcd)

            acc += log_b - log_f
            return (q_new, acc, key), None

        (q_K, acc_K, _), _ = jax.lax.scan(step, (q0, 0.0, key), jnp.arange(K_cmcd))
        return log_g0(q0) - log_g1(q_K) - acc_K

    works = batched_estimate(
        compute_work_single,
        n_samples=cfg.n_samples,
        seed=cfg.seed + 99,
        eval_batch_size=eval_batch_size,
        desc="CMCD eval",
    )
    return {"works": works, "trajs": None, "name": "CMCD"}
