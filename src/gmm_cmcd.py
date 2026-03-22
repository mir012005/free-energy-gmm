"""
gmm_cmcd.py  —  Controlled Monte Carlo Diffusion  (Section 5.3.3)

train() returns (params, loss_history, meta)
estimate() takes (cfg, params, meta) — batched, no trajectory storage
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
    key, k1, k2 = jax.random.split(key, 3)
    comp  = jax.random.choice(k1, jnp.arange(cfg.gmm0.n_components),
                               p=jnp.array(cfg.gmm0.weights))
    means = jnp.array(cfg.gmm0.means)
    covs  = jnp.array(cfg.gmm0.covs)
    q0    = means[comp] + jax.random.multivariate_normal(
                k2, jnp.zeros(d), covs[comp])
    return q0, key


def train(
    cfg: PipelineConfig,
    n_epochs: int            = 5000,
    early_stop_patience: int = 500,
    batch_size: int          = 128,
    lr_init: float           = 2e-4,
    lr_peak: float           = 1e-3,
    lr_end: float            = 1e-5,
    emb_dim: int             = 20,
    grad_clip: float         = 0.5,
    K_cmcd: int              = 64,
) -> tuple[dict, list[float], dict]:

    d       = cfg.dim
    T       = cfg.T
    Lp      = 1.0 / T
    dt_cmcd = T / K_cmcd
    sched   = jnp.linspace(0.0, 1.0, K_cmcd + 1)
    meta    = {"K": K_cmcd, "dt": dt_cmcd}

    V, grad_V, _, log_g0, log_g1 = make_jax_potential(cfg.gmm0, cfg.gmm1)
    init_fn, apply_fn = build_score_network(d, K_cmcd, emb_dim)

    key = jax.random.PRNGKey(cfg.seed)
    key, sk = jax.random.split(key)
    _, params = init_fn(sk)

    def elbo_single(q0, params, key):
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

        (q_K, acc_K, _), _ = jax.lax.scan(
            step, (q0, 0.0, key), jnp.arange(K_cmcd))
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
    optimizer = optax.chain(
        optax.clip_by_global_norm(grad_clip),
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
    cfg: PipelineConfig,
    params: dict,
    meta: dict,
    emb_dim: int         = 20,
    eval_batch_size: int = 256,
) -> dict:
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

        (q_K, acc_K, _), _ = jax.lax.scan(
            step, (q0, 0.0, key), jnp.arange(K_cmcd))
        return log_g0(q0) - log_g1(q_K) - acc_K

    works = batched_estimate(
        compute_work_single,
        n_samples=cfg.n_samples,
        seed=cfg.seed + 99,
        eval_batch_size=eval_batch_size,
        desc="CMCD eval",
    )
    return {"works": works, "trajs": None, "name": "CMCD"}
