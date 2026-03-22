"""
gmm_mcd.py  —  Monte Carlo Diffusion  (Section 5.3.2)

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


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train(
    cfg: PipelineConfig,
    n_epochs: int            = 5000,
    early_stop_patience: int = 500,
    batch_size: int          = 128,
    lr_init: float           = 1e-3,
    lr_peak: float           = 5e-3,
    lr_end: float            = 5e-5,
    emb_dim: int             = 20,
    K_mcd: int               = 64,
) -> tuple[dict, list[float], dict]:

    d      = cfg.dim
    T      = cfg.T
    dt_mcd = T / K_mcd
    sched  = jnp.linspace(0.0, 1.0, K_mcd + 1)
    meta   = {"K": K_mcd, "dt": dt_mcd}

    V, grad_V, _, log_g0, log_g1 = make_jax_potential(cfg.gmm0, cfg.gmm1)
    init_fn, apply_fn = build_score_network(d, K_mcd, emb_dim)

    key = jax.random.PRNGKey(cfg.seed)
    key, sk = jax.random.split(key)
    _, params = init_fn(sk)

    def elbo_single(q0, params, key):
        def step(carry, k):
            q, acc, key = carry
            lam_kp1 = sched[k + 1]
            key, sk = jax.random.split(key)
            noise   = jax.random.normal(sk, (d,))
            gq      = grad_V(q, lam_kp1)
            q_new   = q - dt_mcd * gq + jnp.sqrt(2.0 * dt_mcd) * noise

            diff_f = q_new - q + dt_mcd * gq
            log_f  = -(diff_f ** 2).sum() / (4.0 * dt_mcd)

            score  = apply_fn(params, q_new, k)
            mean_b = (q_new + dt_mcd * grad_V(q_new, lam_kp1)
                            + 2.0 * dt_mcd * score)
            diff_b = q - mean_b
            log_b  = -(diff_b ** 2).sum() / (4.0 * dt_mcd)

            acc += log_b - log_f
            return (q_new, acc, key), None

        (q_K, acc_K, _), _ = jax.lax.scan(
            step, (q0, 0.0, key), jnp.arange(K_mcd))
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
    optimizer = optax.adam(lr_sched)
    opt_state = optimizer.init(params)
    grad_fn   = jax.jit(jax.grad(loss_fn, argnums=1, has_aux=True))

    loss_history, best_loss, patience_ctr = [], float("inf"), 0
    key = jax.random.PRNGKey(cfg.seed + 1)

    pbar = tqdm(range(n_epochs), desc="MCD training")
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
            pbar.set_description(f"MCD [early stop @ {epoch+1}]")
            break

    return params, loss_history, meta


# ─────────────────────────────────────────────────────────────────────────────
# Inference — batched, no trajectory storage
# ─────────────────────────────────────────────────────────────────────────────

def estimate(
    cfg: PipelineConfig,
    params: dict,
    meta: dict,
    emb_dim: int      = 20,
    eval_batch_size: int = 256,
) -> dict:
    d        = cfg.dim
    K        = cfg.n_steps
    dt       = cfg.dt
    sched    = jnp.array(cfg.schedule)
    K_mcd    = meta["K"]
    dt_mcd   = meta["dt"]
    dt_ratio = dt / dt_mcd

    V, grad_V, _, log_g0, log_g1 = make_jax_potential(cfg.gmm0, cfg.gmm1)
    _, apply_fn = build_score_network(d, K_mcd, emb_dim)

    def compute_work_single(seed_val):
        """One trajectory → one scalar work W."""
        key = jax.random.PRNGKey(seed_val)
        q0, key = _sample_q0(cfg, d, key)

        def step(carry, k):
            q, acc, key = carry
            lam_kp1 = sched[k + 1]
            key, sk = jax.random.split(key)
            noise   = jax.random.normal(sk, (d,))
            gq      = grad_V(q, lam_kp1)
            q_new   = q - dt * gq + jnp.sqrt(2.0 * dt) * noise

            diff_f  = q_new - q + dt * gq
            log_f   = -(diff_f ** 2).sum() / (4.0 * dt)

            k_emb   = jnp.clip(
                jnp.floor(k * dt_ratio).astype(int), 0, K_mcd - 1)
            score   = apply_fn(params, q_new, k_emb)
            mean_b  = (q_new + dt * grad_V(q_new, lam_kp1)
                              + 2.0 * dt * score)
            diff_b  = q - mean_b
            log_b   = -(diff_b ** 2).sum() / (4.0 * dt)

            acc += log_b - log_f
            return (q_new, acc, key), None   # no trajectory storage

        (q_K, acc_K, _), _ = jax.lax.scan(
            step, (q0, 0.0, key), jnp.arange(K))
        return log_g0(q0) - log_g1(q_K) - acc_K

    works = batched_estimate(
        compute_work_single,
        n_samples=cfg.n_samples,
        seed=cfg.seed + 99,
        eval_batch_size=eval_batch_size,
        desc="MCD eval",
    )
    return {"works": works, "trajs": None, "name": "MCD"}
