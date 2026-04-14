"""
gmm_jax_physics.py
──────────────────
Pure JAX versions of the GMM energy functions.
Used by MCD, CMCD and LED — all JAX-based algorithms.

CRITICAL: log_g0 / log_g1 return UNNORMALISED log densities
══════════════════════════════════════════════════════════════
The thesis importance weight formula (eq 5.18) uses γ (unnormalised):

    IW = γ_1(q_K)/γ_0(q_0) · Π_k B_k/F_{k+1}

γ_k(q) = Σ_j w_j · exp(-½ (q-μ_j)ᵀ Σ_j⁻¹ (q-μ_j))
        = normalisation_constant_k · π_k(q)

If we used the NORMALISED density log π_k instead of log γ_k, we would get:

    log γ_0(q_0) - log γ_1(q_K)  ←  correct
    log π_0(q_0) - log π_1(q_K)  =  correct - log Z_0 + log Z_1
                                  =  correct + log(Z_1/Z_0)
                                  =  correct - ΔF

Then:  mean(exp(-W_wrong)) = mean(exp(-W_correct + ΔF))
                           = exp(ΔF) · (Z_1/Z_0)
                           = exp(ΔF) · exp(-ΔF)
                           = 1.0   ← this is exactly the bug we observed

By dropping the normalisation constants (which are Z_0 and Z_1 — the very
quantities we are trying to estimate), the boundary terms become correct.

The normalisation constants also cancel exactly in log B_k - log F_{k+1}
since both kernels have the same variance 2dt·I, so the -d/2·log(4πdt)
terms cancel. Hence dropping them from γ is both necessary and consistent.
"""

from __future__ import annotations
import jax
import jax.numpy as jnp
from gmm_config import GMMParams


def make_jax_potential(gmm0: GMMParams, gmm1: GMMParams):
    """
    Returns JAX-jitted energy functions.

    Returns
    -------
    V        : (q, lam) -> scalar    V_λ(q) = -log γ_λ(q)  (unnormalised)
    grad_V   : (q, lam) -> (d,)     ∇_q V_λ(q)
    dV_dlam  : (q, lam) -> scalar   ∂V_λ/∂λ = log γ_0(q) - log γ_1(q)
    log_g0   : (q)      -> scalar   log γ_0(q)  UNNORMALISED
    log_g1   : (q)      -> scalar   log γ_1(q)  UNNORMALISED
    """
    means0   = jnp.array(gmm0.means)    # (K0, d)
    covs0    = jnp.array(gmm0.covs)     # (K0, d, d)
    weights0 = jnp.array(gmm0.weights)  # (K0,)
    means1   = jnp.array(gmm1.means)
    covs1    = jnp.array(gmm1.covs)
    weights1 = jnp.array(gmm1.weights)

    def _log_gmm_unnorm(q, means, covs, weights):
        """
        log γ(q) = log Σ_k w_k exp(-½ (q-μ_k)ᵀ Σ_k⁻¹ (q-μ_k))

        UNNORMALISED: does NOT include -d/2·log(2π) - ½·log|Σ|.
        These are the partition function constants Z_k — exactly what
        we are estimating, so we must not bake them in here.
        """
        def _log_component(mean, cov, w):
            diff = q - mean
            prec = jnp.linalg.inv(cov)
            mah  = diff @ prec @ diff          # (q-μ)ᵀ Σ⁻¹ (q-μ)
            return jnp.log(w + 1e-300) - 0.5 * mah

        log_probs = jax.vmap(_log_component)(means, covs, weights)  # (K,)
        return jax.scipy.special.logsumexp(log_probs)

    def log_g0(q):
        return _log_gmm_unnorm(q, means0, covs0, weights0)

    def log_g1(q):
        return _log_gmm_unnorm(q, means1, covs1, weights1)

    def V(q, lam):
        """V_λ(q) = -[(1-λ) log γ_0(q) + λ log γ_1(q)]"""
        return -(1.0 - lam) * log_g0(q) - lam * log_g1(q)

    def dV_dlam(q, lam):
        """∂V_λ/∂λ = log γ_0(q) - log γ_1(q)"""
        return log_g0(q) - log_g1(q)

    grad_V = jax.grad(V, argnums=0)

    return (
        jax.jit(V),
        jax.jit(grad_V),
        jax.jit(dV_dlam),
        jax.jit(log_g0),
        jax.jit(log_g1),
    )
