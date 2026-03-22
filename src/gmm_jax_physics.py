"""
gmm_jax_physics.py
==================
JAX versions of the energy functions — used by MCD, CMCD and LED.

WHY JAX INSTEAD OF NUMPY?
--------------------------
The three learning algorithms (MCD, CMCD, LED) need to:
    1. Differentiate through the energy functions to compute gradients
       for the Langevin dynamics.
    2. Compile the computation into fast machine code (JIT compilation).
    3. Vectorise over many trajectories simultaneously (vmap).

JAX provides all three capabilities, whereas NumPy does not.

The physics is identical to gmm_physics.py — only the implementation
framework changes.

CRITICAL DETAIL: UNNORMALISED LOG-DENSITIES
============================================
This is perhaps the most important implementation detail in the entire project.

The work formula (eq. 5.18 in the thesis) involves the ratio:

    IW = gamma_1(q_K) / gamma_0(q_0)  *  product of kernel ratios

where gamma_k is the UNNORMALISED density:

    gamma_k(q) = sum_j  w_j * exp(-1/2 * (q-mu_j)^T Sigma_j^{-1} (q-mu_j))

This differs from the normalised density pi_k by a constant Z_k:

    gamma_k(q) = Z_k * pi_k(q)

where Z_k = integral of gamma_k(q) dq  is the partition function.

If we naively used the normalised log-density log pi_k(q) instead of
log gamma_k(q) in the boundary term, we would get:

    log gamma_0(q_0) - log gamma_1(q_K)      <- correct
    log pi_0(q_0)    - log pi_1(q_K)         <- includes -log Z_0 + log Z_1
                                              = correct + log(Z_1/Z_0)
                                              = correct - Delta_F

This would make:
    mean(exp(-W_wrong)) = mean(exp(-W_correct + Delta_F))
                        = exp(Delta_F) * mean(exp(-W_correct))
                        = exp(Delta_F) * (Z_1/Z_0)
                        = exp(Delta_F) * exp(-Delta_F)
                        = 1.0   <-- WRONG! Should be Z_1/Z_0

This exact bug was observed and fixed during development. The fix:
drop the normalisation constants -d/2 * log(2*pi) - 1/2 * log|Sigma|
from the log-density computation.

This is correct because:
    - We are estimating Z_0 and Z_1 — we cannot use them as inputs.
    - The kernel ratio log B_k - log F_{k+1} is unaffected: both kernels
      have variance 2*dt*I, so the normalisation terms cancel exactly.
"""

from __future__ import annotations
import jax
import jax.numpy as jnp
from gmm_config import GMMParams


def make_jax_potential(gmm0: GMMParams, gmm1: GMMParams):
    """
    Build all energy functions needed for MCD, CMCD and LED.

    This function takes the two distributions and returns five JAX-compiled
    functions that can be used inside jax.lax.scan loops and differentiated
    automatically.

    All returned functions operate on a SINGLE particle q of shape (d,).
    Batching over multiple particles is done externally with jax.vmap.

    Parameters
    ----------
    gmm0 : GMMParams — the starting distribution pi_0
    gmm1 : GMMParams — the ending distribution pi_1

    Returns
    -------
    V        : (q, lam) -> scalar
               The potential energy V_lambda(q) = -log gamma_lambda(q).
               This is what the Langevin particle "feels" as a force.

    grad_V   : (q, lam) -> array of shape (d,)
               The gradient of V with respect to q.
               Computed automatically by JAX from V using autodiff.
               This is the "force" pushing the particle.

    dV_dlam  : (q, lam) -> scalar
               The partial derivative dV/d_lambda = log gamma_0(q) - log gamma_1(q).
               Used to accumulate the work W along a trajectory.

    log_g0   : (q,) -> scalar
               Unnormalised log-density of pi_0 at q.
               Used in the boundary terms of the work formula.

    log_g1   : (q,) -> scalar
               Unnormalised log-density of pi_1 at q.
               Used in the boundary terms of the work formula.

    Implementation note
    -------------------
    The function uses Python closures: the returned functions "capture"
    the arrays means0, covs0, weights0, means1, covs1, weights1 from the
    outer scope. This means they remember the GMM parameters without
    needing them as explicit arguments.

    jax.jit(fn) compiles fn the first time it is called and caches the
    compiled version — subsequent calls are much faster.
    """

    # Convert GMM parameters from NumPy to JAX arrays once.
    # This happens at Python level (not inside JAX's computation graph)
    # so it does not slow down the compiled functions.
    means0   = jnp.array(gmm0.means)    # shape (K0, d)
    covs0    = jnp.array(gmm0.covs)     # shape (K0, d, d)
    weights0 = jnp.array(gmm0.weights)  # shape (K0,)
    means1   = jnp.array(gmm1.means)
    covs1    = jnp.array(gmm1.covs)
    weights1 = jnp.array(gmm1.weights)

    def _log_gmm_unnorm(q, means, covs, weights):
        """
        Compute the UNNORMALISED log-density of a GMM at point q.

        FORMULA:
            log gamma(q) = log [ sum_k  w_k * exp(-1/2 * (q-mu_k)^T Sigma_k^{-1} (q-mu_k)) ]

        WHAT WE INTENTIONALLY OMIT (and why):
            The full normalised log-density would include -d/2 * log(2*pi) - 1/2 * log|Sigma_k|
            for each component. We drop these terms because they are exactly log(Z_k)
            — the partition functions we are trying to estimate.
            Including them would bias the estimator (see the file-level docstring).

        HOW IT IS COMPUTED:
            For each component k:
                log_k = log(w_k) - 1/2 * (q - mu_k)^T Sigma_k^{-1} (q - mu_k)

            The squared term (q-mu_k)^T Sigma_k^{-1} (q-mu_k) is called the
            "Mahalanobis distance" — it measures how far q is from the centre
            mu_k, normalised by the shape of the Gaussian.

            Then we combine the K terms using the log-sum-exp operation:
                log gamma(q) = log(sum_k exp(log_k))

            log-sum-exp is used instead of directly computing sum(exp(...))
            to avoid numerical overflow when the exponents are large.

        Parameters
        ----------
        q       : array of shape (d,) — the point to evaluate
        means   : array of shape (K, d)
        covs    : array of shape (K, d, d)
        weights : array of shape (K,)

        Returns
        -------
        scalar — log gamma(q)
        """
        def _log_component_unnorm(mean, cov, w):
            """Compute log w_k - 1/2 * Mahalanobis_k(q) for one component."""
            diff = q - mean                      # displacement from centre: shape (d,)
            prec = jnp.linalg.inv(cov)           # precision matrix Sigma^{-1}: shape (d, d)
            mah  = diff @ prec @ diff            # Mahalanobis distance: scalar
            return jnp.log(w + 1e-300) - 0.5 * mah
            # 1e-300 prevents log(0) when a weight is exactly zero

        # Apply _log_component_unnorm to all K components at once using vmap.
        # jax.vmap vectorises a function over its first argument(s).
        # Here it maps over the first axis of means, covs, weights (i.e., over k).
        log_probs = jax.vmap(_log_component_unnorm)(means, covs, weights)  # shape (K,)

        # Combine with log-sum-exp: log(sum_k exp(log_probs_k))
        return jax.scipy.special.logsumexp(log_probs)   # scalar

    # --- Build the five returned functions ---

    def log_g0(q):
        """Unnormalised log-density of pi_0 at q. See _log_gmm_unnorm."""
        return _log_gmm_unnorm(q, means0, covs0, weights0)

    def log_g1(q):
        """Unnormalised log-density of pi_1 at q. See _log_gmm_unnorm."""
        return _log_gmm_unnorm(q, means1, covs1, weights1)

    def V(q, lam):
        """
        Interpolated potential energy:

            V_lambda(q) = -[(1-lambda) * log gamma_0(q) + lambda * log gamma_1(q)]

        At lambda=0: V = -log gamma_0(q)  ->  particle feels pi_0
        At lambda=1: V = -log gamma_1(q)  ->  particle feels pi_1
        """
        return -(1.0 - lam) * log_g0(q) - lam * log_g1(q)

    def dV_dlam(q, lam):
        """
        Partial derivative of V with respect to lambda:

            dV/d_lambda = log gamma_0(q) - log gamma_1(q)

        This is the instantaneous work rate: as lambda increases by d_lambda,
        the work done on the particle is dV/d_lambda * d_lambda.

        Note: this does not depend on lambda (the formula has no lambda term).
        The lam argument is kept for API consistency only.
        """
        return log_g0(q) - log_g1(q)

    # grad_V is computed automatically by JAX's autodiff.
    # jax.grad(V, argnums=0) returns a function that computes
    # the gradient of V with respect to its first argument (q).
    # This saves us from computing the gradient by hand.
    grad_V = jax.grad(V, argnums=0)

    # jax.jit compiles each function with XLA for fast execution.
    # The first call will be slow (compilation), subsequent calls are fast.
    return (
        jax.jit(V),
        jax.jit(grad_V),
        jax.jit(dV_dlam),
        jax.jit(log_g0),
        jax.jit(log_g1),
    )
