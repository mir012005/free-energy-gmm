"""
gmm_led.py  —  Learning an Escorting Drift  (Section 5.2.2)

train() returns (params, loss_history, meta)
estimate() takes (cfg, params, meta) — batched, no trajectory storage

loss_type options:
  "work_variance" : Var(W^theta) -> 0 at optimum.
                    Avoids W->0 degeneracy but does not anchor E[W] to Delta_F.
                    Result: low variance but biased ratio estimator.

  "log_variance"  : log(Var(W^theta)) -> -inf at optimum.  [reference 256]
                    More numerically stable than raw variance.
                    Same theoretical properties as work_variance.

  "work_mean"     : E[W^theta] -> Delta_F at optimum in theory.
                    But network can achieve loss=0 via degenerate drift.
                    Use only for comparison / diagnosis.
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


def _hutchinson_div(fn, q, key, n_probes=4):
    def one(sk):
        v = jax.random.rademacher(sk, shape=q.shape, dtype=q.dtype)
        _, vjp_fn = jax.vjp(fn, q)
        return (vjp_fn(v)[0] * v).sum()
    return jax.vmap(one)(jax.random.split(key, n_probes)).mean()


def _exact_div(fn, q):
    return jnp.trace(jax.jacfwd(fn)(q))


def train(
    cfg: PipelineConfig,
    n_epochs: int            = 5000,
    early_stop_patience: int = 500,
    batch_size: int          = 128,
    lr_init: float           = 1e-3,
    lr_peak: float           = 5e-3,
    lr_end: float            = 5e-5,
    emb_dim: int             = 20,
    exact_div: bool          = False,
    n_probes: int            = 4,
    grad_clip: float         = 1.0,
    loss_type: str           = "log_variance",  # best default
    K_led: int               = 64,
) -> tuple[dict, list[float], dict]:

    d      = cfg.dim
    T      = cfg.T
    Lp     = 1.0 / T
    dt_led = T / K_led
    sched  = jnp.linspace(0.0, 1.0, K_led + 1)
    meta   = {"K": K_led, "dt": dt_led}

    V, grad_V, dV_dlam, log_g0, log_g1 = make_jax_potential(cfg.gmm0, cfg.gmm1)
    init_fn, apply_fn = build_score_network(d, K_led, emb_dim)

    key = jax.random.PRNGKey(cfg.seed)
    key, sk = jax.random.split(key)
    _, params = init_fn(sk)

    def work_single(q0, params, key):
        def step(carry, k):
            q, w, key = carry
            lam_k   = sched[k]
            lam_kp1 = sched[k + 1]
            dlam    = lam_kp1 - lam_k

            key, div_key, step_key = jax.random.split(key, 3)
            u_fn  = lambda qq: apply_fn(params, qq, k)
            u_q   = u_fn(q)
            div_u = (_exact_div(u_fn, q) if exact_div
                     else _hutchinson_div(u_fn, q, div_key, n_probes))

            dVdl = dV_dlam(q, lam_k)
            gq   = grad_V(q, lam_k)
            w   += dlam * (dVdl + (u_q * gq).sum() - div_u)

            noise = jax.random.normal(step_key, (d,))
            q_new = (q - dt_led * grad_V(q, lam_kp1)
                       + dt_led * Lp * u_q
                       + jnp.sqrt(2.0 * dt_led) * noise)
            return (q_new, w, key), None

        (_, w_K, _), _ = jax.lax.scan(
            step, (q0, 0.0, key), jnp.arange(K_led))
        return w_K

    def loss_fn(seeds, params):
        def one(seed):
            k = jax.random.PRNGKey(seed)
            q0, k = _sample_q0(cfg, d, k)
            return work_single(q0, params, k)
        works = jax.vmap(one)(seeds)

        if loss_type == "log_variance":
            # log(Var(W)) — more stable than raw Var, same minimum at Var=0
            loss = jnp.log(works.var() + 1e-8)
        elif loss_type == "work_variance":
            loss = works.var()
        else:
            # "work_mean" — E[W], has W->0 degeneracy
            loss = works.mean()

        return loss, (loss, works.mean())

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

    pbar = tqdm(range(n_epochs), desc=f"LED [{loss_type}]")
    for epoch in pbar:
        key, sk = jax.random.split(key)
        seeds   = jax.random.randint(sk, (batch_size,), 1, int(1e6))
        grads, (loss, work_mean) = grad_fn(seeds, params)
        loss_val  = float(loss)
        wmean_val = float(work_mean)
        loss_history.append(wmean_val)   # always log E[W] for interpretability
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        if loss_val < best_loss:
            best_loss, patience_ctr = loss_val, 0
        else:
            patience_ctr += 1
        if patience_ctr >= early_stop_patience:
            pbar.set_description(f"LED [early stop @ {epoch+1}]")
            break

    return params, loss_history, meta


def estimate(
    cfg: PipelineConfig,
    params: dict,
    meta: dict,
    emb_dim: int         = 20,
    exact_div: bool      = False,
    n_probes: int        = 4,
    eval_batch_size: int = 256,
) -> dict:
    d      = cfg.dim
    T      = cfg.T
    Lp     = 1.0 / T
    K_led  = meta["K"]
    dt_led = meta["dt"]
    sched  = jnp.linspace(0.0, 1.0, K_led + 1)

    V, grad_V, dV_dlam, log_g0, log_g1 = make_jax_potential(cfg.gmm0, cfg.gmm1)
    _, apply_fn = build_score_network(d, K_led, emb_dim)

    def compute_work_single(seed_val):
        key = jax.random.PRNGKey(seed_val)
        q0, key = _sample_q0(cfg, d, key)

        def step(carry, k):
            q, w, key = carry
            lam_k   = sched[k]
            lam_kp1 = sched[k + 1]
            dlam    = lam_kp1 - lam_k

            key, div_key, step_key = jax.random.split(key, 3)
            u_fn  = lambda qq: apply_fn(params, qq, k)
            u_q   = u_fn(q)
            div_u = (_exact_div(u_fn, q) if exact_div
                     else _hutchinson_div(u_fn, q, div_key, n_probes))

            dVdl = dV_dlam(q, lam_k)
            gq   = grad_V(q, lam_k)
            w   += dlam * (dVdl + (u_q * gq).sum() - div_u)

            noise = jax.random.normal(step_key, (d,))
            q_new = (q - dt_led * grad_V(q, lam_kp1)
                       + dt_led * Lp * u_q
                       + jnp.sqrt(2.0 * dt_led) * noise)
            return (q_new, w, key), None

        (_, w_K, _), _ = jax.lax.scan(
            step, (q0, 0.0, key), jnp.arange(K_led))
        return w_K

    works = batched_estimate(
        compute_work_single,
        n_samples=cfg.n_samples,
        seed=cfg.seed + 99,
        eval_batch_size=eval_batch_size,
        desc="LED eval",
    )
    return {"works": works, "trajs": None, "name": "LED"}
