"""
gmm_led.py
==========
Learning an Escorting Drift (LED) — Section 5.2.2 of the thesis.

DIFFERENT APPROACH FROM MCD AND CMCD
--------------------------------------
MCD and CMCD both work with DISCRETE Langevin chains and optimise
a path-space ELBO involving log-ratios of transition kernels.

LED takes a fundamentally different approach rooted in the theory of
stochastic differential equations (SDEs) and optimal transport.

CONTINUOUS-TIME WORK FORMULA (eq. 5.5 / 5.11)
-----------------------------------------------
Consider the CONTROLLED SDE:

    dq = [-grad V_{Lambda(s)}(q)  +  Lambda'(s) * u_theta(q, s)] ds  +  sqrt(2) dW

where:
    Lambda(s) : the switching schedule, Lambda(0)=0, Lambda(T)=1
    Lambda'(s) : its derivative = 1/T for linear scheduling
    u_theta    : the learned escorting drift (a neural network)
    dW         : Brownian motion noise

The work accumulated along a trajectory of this SDE is (eq. 5.11):

    W^theta = integral_0^T Lambda'(s) * [dV/d_lambda(q_s)
                                          + u_theta(q_s, s) . grad V_{Lambda(s)}(q_s)
                                          - div(u_theta)(q_s, s)] ds

where div(u_theta) = sum_i d u_i / d q_i  is the divergence of u_theta.

JARZYNSKI EQUALITY STILL HOLDS:
    E[exp(-W^theta)] = Z1/Z0  for ANY drift u_theta.

THE OPTIMAL DRIFT
-----------------
The optimal drift u^* minimises the variance of exp(-W^theta).
At the optimum:
    - W^theta = Delta_F for EVERY trajectory (zero variance!)
    - All trajectories give the same work regardless of fluctuations

Finding u^* is equivalent to solving a Poisson equation related to the
generator of the SDE. This is discussed in Section 5.5 of the thesis.

TRAINING OBJECTIVE
------------------
We minimise the log-variance loss (eq. 5.14):

    L(theta) = log(Var(W^theta)) = log(E[(W^theta)^2] - (E[W^theta])^2)

At the optimum: Var(W^theta) = 0, so L -> -infinity.

WHY LOG-VARIANCE INSTEAD OF VARIANCE?
The raw variance loss Var(W^theta) has very small gradients when Var is
already small (flat landscape). The log makes the loss more sensitive near
the optimum, improving convergence.

WHY NOT E[W^theta]?
E[W^theta] should converge to Delta_F at the optimum. But the network can
"cheat" by making u_theta such that the divergence term cancels dV/dlambda
everywhere, giving W^theta = 0 for all trajectories. This satisfies E[W]=0
but exp(-W) = 1 != Z1/Z0. The variance loss avoids this degeneracy because
Var(W)=0 only when W = constant = Delta_F for all trajectories.

DIVERGENCE COMPUTATION
-----------------------
The divergence div(u_theta) = Tr(Jacobian(u_theta)) requires computing
the trace of the d x d Jacobian matrix. Two methods:

    EXACT: div = Tr(J) via jax.jacfwd (full Jacobian)
           Cost: O(d) forward passes
           Use for small d (d <= 5)

    HUTCHINSON: div ≈ E_v[v^T J v] where v ~ Rademacher(+/-1)
           Cost: O(n_probes) vector-Jacobian products
           Unbiased estimator, O(1/n_probes) variance
           Use for large d (d > 5) to avoid OOM
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


def _hutchinson_div(fn, q, key, n_probes=4):
    """
    Estimate the divergence div(fn)(q) = sum_i d fn_i / d q_i
    using the Hutchinson stochastic trace estimator.

    IDEA (Hutchinson 1989):
        For any matrix A: Tr(A) = E_v[v^T A v]  where v ~ Rademacher(+/-1)

    Applied to the Jacobian J of fn at q:
        Tr(J) = E_v[v^T J v] = E_v[(J^T v) . v]

    We estimate this expectation with n_probes samples:
        Tr(J) ≈ (1/n_probes) * sum_i v_i^T J v_i

    Each v_i^T J is computed via a single vector-Jacobian product (vjp),
    which costs the same as one backward pass through fn.

    Parameters
    ----------
    fn      : callable q -> R^d, the function whose divergence we want
    q       : array of shape (d,) — the point at which to evaluate
    key     : JAX random key for generating Rademacher vectors
    n_probes : int — number of random probes (more = lower variance)

    Returns
    -------
    scalar — the Hutchinson estimate of div(fn)(q)
    """
    def one(sk):
        # Sample one Rademacher vector: each component is +1 or -1 with equal prob
        v = jax.random.rademacher(sk, shape=q.shape, dtype=q.dtype)
        # Compute v^T J via vector-Jacobian product: (J^T v)_i = sum_j v_j * (d fn_j / d q_i)
        _, vjp_fn = jax.vjp(fn, q)
        # vjp_fn(v) = J^T v, shape (d,)
        # v . (J^T v) = v^T J^T v = (Jv).v = v^T J v (since Tr(A) = Tr(A^T))
        return (vjp_fn(v)[0] * v).sum()

    # Average over n_probes independent samples
    return jax.vmap(one)(jax.random.split(key, n_probes)).mean()


def _exact_div(fn, q):
    """
    Compute the exact divergence div(fn)(q) = Tr(Jacobian(fn)(q)).

    This computes the full d x d Jacobian matrix using forward-mode
    automatic differentiation (jax.jacfwd), then takes its trace.

    Cost: O(d) forward passes through fn.
    Only use for small d (d <= 5 or so).

    Parameters
    ----------
    fn : callable q -> R^d
    q  : array of shape (d,)

    Returns
    -------
    scalar — the exact divergence at q
    """
    return jnp.trace(jax.jacfwd(fn)(q))
    # jax.jacfwd computes the Jacobian using forward-mode AD
    # jnp.trace takes the sum of diagonal elements


def train(
    cfg:                  PipelineConfig,
    n_epochs:             int   = 5000,
    early_stop_patience:  int   = 500,
    batch_size:           int   = 128,
    lr_init:              float = 1e-3,
    lr_peak:              float = 5e-3,
    lr_end:               float = 5e-5,
    emb_dim:              int   = 20,
    exact_div:            bool  = False,
    n_probes:             int   = 4,
    grad_clip:            float = 1.0,
    loss_type:            str   = "log_variance",
    K_led:                int   = 64,
) -> tuple[dict, list[float], dict]:
    """
    Train the LED escorting drift network u_theta.

    Parameters
    ----------
    cfg : PipelineConfig
    n_epochs : int
    early_stop_patience : int
    batch_size : int
    lr_init, lr_peak, lr_end : float — learning rate schedule
    emb_dim : int — time embedding dimension
    exact_div : bool — if True, use exact Jacobian trace (only for d <= 5)
                       if False, use Hutchinson estimator (recommended)
    n_probes : int — number of Rademacher probes for Hutchinson estimator
                     More probes = lower variance estimate of divergence.
    grad_clip : float — maximum gradient norm
    loss_type : str — one of:
        "log_variance"  : log(Var(W^theta)) [recommended, more stable]
        "work_variance" : Var(W^theta) [same minimum, less stable near 0]
        "work_mean"     : E[W^theta] [has degeneracy W->0, not recommended]
    K_led : int — number of Langevin steps for training

    Returns
    -------
    params : dict — trained network parameters
    loss_history : list — E[W^theta] at each epoch (for comparison across methods)
    meta : dict — {"K": K_led, "dt": dt_led}
    """
    d      = cfg.dim
    T      = cfg.T
    Lp     = 1.0 / T           # Lambda'(s) = 1/T
    dt_led = T / K_led
    sched  = jnp.linspace(0.0, 1.0, K_led + 1)
    meta   = {"K": K_led, "dt": dt_led}

    V, grad_V, dV_dlam, log_g0, log_g1 = make_jax_potential(cfg.gmm0, cfg.gmm1)
    init_fn, apply_fn = build_score_network(d, K_led, emb_dim)

    key = jax.random.PRNGKey(cfg.seed)
    key, sk = jax.random.split(key)
    _, params = init_fn(sk)

    def work_single(q0, params, key):
        """
        Simulate one ESCORTED trajectory and compute the work W^theta.

        The work is accumulated using the discrete version of eq. 5.14:

            W^theta = sum_{k=0}^{K-1} (lambda_{k+1} - lambda_k)
                      * [dV/dlambda(q_k) + u_theta(q_k) . grad V(q_k) - div(u_theta)(q_k)]

        Each term has a physical meaning:
            dV/dlambda  : the "standard" AIS work contribution (also done by AIS)
            u . grad V  : the drift increases work when aligned with the force
            -div(u)     : the divergence correction (from Ito's formula / Girsanov)
                          ensures Jarzynski's equality still holds

        Parameters
        ----------
        q0     : array of shape (d,)
        params : network parameters
        key    : JAX random key

        Returns
        -------
        scalar — the total escorted work W^theta for this trajectory
        """
        def step(carry, k):
            q, w, key = carry
            lam_k   = sched[k]
            lam_kp1 = sched[k + 1]
            dlam    = lam_kp1 - lam_k   # lambda increment

            # Split key for divergence estimation and EM step
            key, div_key, step_key = jax.random.split(key, 3)

            # Define u_fn as a function of position only (fixing step index k)
            # This is needed for computing the Jacobian with respect to q
            u_fn = lambda qq: apply_fn(params, qq, k)
            u_q  = u_fn(q)   # the drift at current position, shape (d,)

            # Compute divergence of u_theta at q
            if exact_div:
                div_u = _exact_div(u_fn, q)
            else:
                div_u = _hutchinson_div(u_fn, q, div_key, n_probes)

            # Work contribution at step k (discrete Clausius work with correction)
            dVdl = dV_dlam(q, lam_k)    # dV/dlambda = log gamma_0 - log gamma_1
            gq   = grad_V(q, lam_k)     # grad V at current lambda
            w   += dlam * (dVdl + (u_q * gq).sum() - div_u)
            # Note: we use lam_k (not lam_kp1) for the gradient in the work formula
            # This is the left-Riemann sum approximation of the integral

            # Escorted EM step: move to next position
            noise = jax.random.normal(step_key, (d,))
            q_new = (q
                     - dt_led * grad_V(q, lam_kp1)   # standard Langevin drift at lam_{k+1}
                     + dt_led * Lp * u_q              # escorting drift
                     + jnp.sqrt(2.0 * dt_led) * noise)

            return (q_new, w, key), None

        (_, w_K, _), _ = jax.lax.scan(step, (q0, 0.0, key), jnp.arange(K_led))
        return w_K

    def loss_fn(seeds, params):
        """
        Compute the training loss over a batch of trajectories.

        Three loss options:
            "log_variance" : log(Var(W)) — smooth near 0, avoids flat landscape
            "work_variance": Var(W)      — direct variance minimisation
            "work_mean"    : E[W]        — has degenerate solution W=0 (avoid)

        We always log E[W] as loss_history for interpretability.
        """
        def one(seed):
            k = jax.random.PRNGKey(seed)
            q0, k = _sample_q0(cfg, d, k)
            return work_single(q0, params, k)

        works = jax.vmap(one)(seeds)   # shape (batch_size,)

        if loss_type == "log_variance":
            # log(Var(W) + epsilon): epsilon=1e-8 prevents log(0)
            loss = jnp.log(works.var() + 1e-8)
        elif loss_type == "work_variance":
            loss = works.var()
        else:
            # "work_mean": E[W] — only for debugging/comparison
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

        # Log E[W] (not the variance loss) for interpretability in plots
        # E[W] should converge toward Delta_F for a well-trained model
        loss_history.append(wmean_val)

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
    cfg:            PipelineConfig,
    params:         dict,
    meta:           dict,
    emb_dim:        int  = 20,
    exact_div:      bool = False,
    n_probes:       int  = 4,
    eval_batch_size: int = 256,
) -> dict:
    """
    Estimate Z1/Z0 using the trained LED escorting drift.

    Simulates N escorted trajectories and computes W^theta for each.
    By Jarzynski's equality: E[exp(-W^theta)] = Z1/Z0.

    Note: LED uses the SAME K_led and dt_led as training (no grid change).

    Parameters
    ----------
    cfg    : PipelineConfig — evaluation configuration
    params : dict — trained network parameters from train()
    meta   : dict — {"K": K_led, "dt": dt_led} from train()
    emb_dim : int — must match training
    exact_div : bool — whether to use exact Jacobian trace
    n_probes : int — number of Hutchinson probes
    eval_batch_size : int — mini-batch size

    Returns
    -------
    dict with keys "works", "trajs" (None), "name" ("LED")
    """
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
            u_fn = lambda qq: apply_fn(params, qq, k)
            u_q  = u_fn(q)
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

        (_, w_K, _), _ = jax.lax.scan(step, (q0, 0.0, key), jnp.arange(K_led))
        return w_K

    works = batched_estimate(
        compute_work_single,
        n_samples=cfg.n_samples,
        seed=cfg.seed + 99,
        eval_batch_size=eval_batch_size,
        desc="LED eval",
    )
    return {"works": works, "trajs": None, "name": "LED"}
