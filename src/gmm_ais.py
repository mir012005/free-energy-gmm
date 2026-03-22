"""
gmm_ais.py  —  Annealed Importance Sampling  (= standard Jarzynski)

Work (eq 5.9 / 5.22):
    W = Σ_{k=0}^{K-1} [V_{λ_{k+1}}(q_k) - V_{λ_k}(q_k)]
      = Σ_{k=0}^{K-1} (λ_{k+1}-λ_k) · [log γ_0(q_k) - log γ_1(q_k)]

Uses JAX + jax.lax.scan for speed, batched evaluation to avoid OOM.
No trajectory storage at eval time.
"""

from __future__ import annotations
import jax
import jax.numpy as jnp
import numpy as np

from gmm_config import PipelineConfig
from gmm_jax_physics import make_jax_potential
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


def run(cfg: PipelineConfig, eval_batch_size: int = 256) -> dict:
    d     = cfg.dim
    K     = cfg.n_steps
    dt    = cfg.dt
    sched = jnp.array(cfg.schedule)

    V, grad_V, dV_dlam, log_g0, log_g1 = make_jax_potential(cfg.gmm0, cfg.gmm1)

    def compute_work_single(seed_val):
        """One AIS trajectory → scalar W."""
        key = jax.random.PRNGKey(seed_val)
        q0, key = _sample_q0(cfg, d, key)

        def step(carry, k):
            q, w, key = carry
            lam_k   = sched[k]
            lam_kp1 = sched[k + 1]
            dlam    = lam_kp1 - lam_k

            # Work: (λ_{k+1}-λ_k) · ∂V/∂λ(q_k) = dlam · (log γ_0 - log γ_1)(q_k)
            w += dlam * dV_dlam(q, lam_k)

            # EM step at λ_{k+1}
            key, sk = jax.random.split(key)
            noise   = jax.random.normal(sk, (d,))
            gq      = grad_V(q, lam_kp1)
            q_new   = q - dt * gq + jnp.sqrt(2.0 * dt) * noise

            return (q_new, w, key), None

        (_, w_K, _), _ = jax.lax.scan(step, (q0, 0.0, key), jnp.arange(K))
        return w_K

    works = batched_estimate(
        compute_work_single,
        n_samples=cfg.n_samples,
        seed=cfg.seed,
        eval_batch_size=eval_batch_size,
        desc="AIS eval",
    )
    return {"works": works, "trajs": None, "name": "AIS"}
