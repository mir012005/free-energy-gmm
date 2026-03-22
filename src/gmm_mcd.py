"""
gmm_mcd.py
==========
Monte Carlo Diffusion (MCD) — Section 5.3.2 of the thesis.

PHYSICAL IDEA
-------------
AIS uses the standard Langevin dynamics to move the particle from pi_0 to
pi_1. The importance weight (virtual work) has high variance because the
forward trajectories are not time-reversible: the path from q_0 to q_K is
very different from the reversed path from q_K to q_0.

MCD improves AIS by learning the OPTIMAL TIME-REVERSAL of the dynamics.
If we know the reverse process exactly, we can construct an importance
weight with much lower variance.

MATHEMATICAL BACKGROUND
------------------------
For a Langevin trajectory q_0 -> q_1 -> ... -> q_K, define:

    Forward kernel:  F_{k+1}(q' | q) = N(q - dt*grad V_{lambda_{k+1}}(q), 2*dt*I)
    This is the transition probability of the EM step: given q_k, what is
    the probability of landing at q_{k+1}?

    Backward kernel: B_k(q | q') = the probability of the reverse step
    In theory: B_k(q | q') proportional to F_{k+1}(q' | q) * p_k(q) / p_{k+1}(q')
    This is the time-reversal of the forward kernel under the stationary measure.

The importance weight (eq. 5.18) is:

    IW = [gamma_1(q_K) / gamma_0(q_0)] * product_{k=0}^{K-1} B_k(q_k | q_{k+1}) / F_{k+1}(q_{k+1} | q_k)

The virtual work is W = -log(IW).

THE PROBLEM: B_k is unknown because it involves the intractable densities
p_k(q) of the interpolated distributions.

MCD's SOLUTION: APPROXIMATE B_k WITH A NEURAL NETWORK
------------------------------------------------------
MCD parametrises the backward kernel as:

    B^theta_k(q | q') = N(q' + dt*grad V_{lambda_{k+1}}(q') + 2*dt*s_theta(q', k), 2*dt*I)

where s_theta(q, k) is a neural network (the "score network").

If s_theta(q, k) = grad_q log p_k(q) (the true score function), then B^theta
matches the true time-reversal exactly and the estimator has minimum variance.

TRAINING OBJECTIVE (ELBO)
--------------------------
The network is trained by maximising the Evidence Lower BOund (ELBO):

    ELBO(theta) = E_Q[ log gamma_1(q_K) - log gamma_0(q_0)
                       + sum_k [log B^theta_k(q_k | q_{k+1}) - log F_{k+1}(q_{k+1} | q_k)] ]

This is equivalent to minimising the KL divergence between the forward
trajectory distribution Q and the backward process. At optimum:
    - s_theta converges to the true score grad log p_k
    - ELBO = -Delta_F
    - The loss (= -ELBO) converges to Delta_F

IMPORTANT: unlike AIS, the trajectories used for training come from the
FORWARD process (no need to simulate backward). The ELBO is maximised by
adjusting only the backward kernel parameters theta.

KEY IMPLEMENTATION DETAILS
---------------------------
1. Fixed trajectory batch: the forward dynamics does NOT depend on theta,
   so we could reuse the same trajectories across training epochs.
   In practice we resample each epoch for stochasticity (acts as data augmentation).

2. Training uses K=64 steps (not K=1000): scanning 1000 steps across 128 trajectories
   in a jax.lax.scan uses too much memory. K=64 with dt=T/64 learns a good
   score while staying within memory bounds.

3. Evaluation uses finer steps: at evaluation time we use dt=1e-3 (K=1000 steps)
   for a more accurate estimate. The trained embedding index is mapped via
   k_eval -> floor(k_eval * dt_eval / dt_train).

4. Early stopping: if the loss does not improve for `early_stop_patience` epochs,
   training is halted to avoid overfitting.
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
    """
    Sample one initial particle position from pi_0.
    See gmm_ais.py for a detailed explanation of this function.
    """
    key, k1, k2 = jax.random.split(key, 3)
    comp  = jax.random.choice(k1, jnp.arange(cfg.gmm0.n_components),
                               p=jnp.array(cfg.gmm0.weights))
    means = jnp.array(cfg.gmm0.means)
    covs  = jnp.array(cfg.gmm0.covs)
    q0    = means[comp] + jax.random.multivariate_normal(k2, jnp.zeros(d), covs[comp])
    return q0, key


# =============================================================================
# Training
# =============================================================================

def train(
    cfg:                  PipelineConfig,
    n_epochs:             int   = 5000,
    early_stop_patience:  int   = 500,
    batch_size:           int   = 128,
    lr_init:              float = 1e-3,
    lr_peak:              float = 5e-3,
    lr_end:               float = 5e-5,
    emb_dim:              int   = 20,
    K_mcd:                int   = 64,
) -> tuple[dict, list[float], dict]:
    """
    Train the MCD score network.

    The training loop:
        For each epoch:
            1. Sample a batch of `batch_size` random seeds
            2. For each seed: simulate one trajectory and compute the ELBO
            3. Average the ELBO over the batch -> the training objective
            4. Compute gradients of -ELBO with respect to network parameters
            5. Update parameters with the Adam optimiser
            6. Check early stopping

    Parameters
    ----------
    cfg : PipelineConfig
        The simulation setup (distributions, time, etc.)

    n_epochs : int
        Maximum number of training epochs.

    early_stop_patience : int
        Stop training if the loss does not improve for this many epochs.
        Prevents wasting time if the network has already converged.

    batch_size : int
        Number of trajectories per training step.
        Larger = more stable gradient estimates, but uses more memory.

    lr_init : float
        Initial learning rate for the warm-up phase.

    lr_peak : float
        Peak learning rate (reached after warm-up).

    lr_end : float
        Final learning rate at the end of training (cosine decay).

    emb_dim : int
        Dimension of the time embedding vectors in the network.

    K_mcd : int
        Number of Langevin steps used during TRAINING.
        Kept small (64) to avoid Out Of Memory errors.
        Evaluation uses more steps via the embedding index mapping.

    Returns
    -------
    params : dict
        Trained network parameters (pure JAX arrays, no Python scalars).
        Save with pickle for reuse.

    loss_history : list of float
        Loss value at each epoch. Useful for plotting convergence.

    meta : dict
        {"K": K_mcd, "dt": dt_mcd}
        Training configuration needed by estimate() to interpret the params.
    """
    d      = cfg.dim
    T      = cfg.T
    dt_mcd = T / K_mcd                           # time step for training
    sched  = jnp.linspace(0.0, 1.0, K_mcd + 1)  # lambda schedule for training
    meta   = {"K": K_mcd, "dt": dt_mcd}

    V, grad_V, _, log_g0, log_g1 = make_jax_potential(cfg.gmm0, cfg.gmm1)

    # Build the score network (see gmm_networks.py for architecture details)
    init_fn, apply_fn = build_score_network(d, K_mcd, emb_dim)

    # Initialise the network parameters with a random key
    key = jax.random.PRNGKey(cfg.seed)
    key, sk = jax.random.split(key)
    _, params = init_fn(sk)
    # params is now a dict of JAX arrays — pure, no Python scalars

    def elbo_single(q0, params, key):
        """
        Compute the ELBO for ONE trajectory starting at q0.

        This function is the core of MCD training. It simulates K steps of
        forward Langevin dynamics and computes:

            ELBO = log gamma_1(q_K) - log gamma_0(q_0)
                   + sum_k [log B^theta_k(q_k | q_{k+1}) - log F_{k+1}(q_{k+1} | q_k)]

        FORWARD KERNEL (eq. 5.24):
            F_{k+1}(q_{k+1} | q_k) = N(q_k - dt * grad V_{lambda_{k+1}}(q_k), 2*dt*I)

            log F = -||q_{k+1} - q_k + dt * grad V||^2 / (4*dt)
            (from the formula for the log of a Gaussian)

        BACKWARD KERNEL (eq. 5.26):
            B^theta_k(q_k | q_{k+1}) = N(q_{k+1} + dt * grad V_{lambda_{k+1}}(q_{k+1})
                                          + 2*dt * s_theta(q_{k+1}, k), 2*dt*I)

            log B = -||q_k - mean_b||^2 / (4*dt)
            where mean_b = q_{k+1} + dt * grad V + 2*dt * s_theta

        At optimum: log B - log F = log[p_k(q_k) / p_{k+1}(q_{k+1})] for each k
        and the sum telescopes to log[p_0(q_0) / p_K(q_K)]
        making the ELBO = log[gamma_1(q_K) / gamma_0(q_0) * p_0(q_0) / p_K(q_K)]

        Parameters
        ----------
        q0     : array of shape (d,) — starting position
        params : network parameters
        key    : JAX random key

        Returns
        -------
        scalar — the ELBO value for this trajectory
        """
        def step(carry, k):
            """
            One step of the MCD forward simulation.
            Computes the work contribution log B - log F for step k.
            """
            q, acc, key = carry
            lam_kp1 = sched[k + 1]   # lambda at next step

            # Generate noise and take one EM step
            key, sk = jax.random.split(key)
            noise   = jax.random.normal(sk, (d,))
            gq      = grad_V(q, lam_kp1)                    # grad V at new lambda
            q_new   = q - dt_mcd * gq + jnp.sqrt(2.0 * dt_mcd) * noise

            # --- log F_{k+1}(q_new | q) ---
            # This is the probability that the EM step took us from q to q_new.
            # diff_f = q_new - (q - dt * grad V)  =  the noise term (scaled)
            diff_f = q_new - q + dt_mcd * gq
            log_f  = -(diff_f ** 2).sum() / (4.0 * dt_mcd)
            # Formula: log N(x; 0, 2*dt*I) = -||x||^2 / (4*dt) + constant
            # The constant (-d/2 * log(4*pi*dt)) cancels in log B - log F.

            # --- log B^theta_k(q | q_new) ---
            # The backward kernel "tries to guess" where q came from, given q_new.
            # The mean of the backward kernel uses the score network:
            score  = apply_fn(params, q_new, k)              # s_theta(q_new, k)
            mean_b = (q_new + dt_mcd * grad_V(q_new, lam_kp1)
                            + 2.0 * dt_mcd * score)
            diff_b = q - mean_b
            log_b  = -(diff_b ** 2).sum() / (4.0 * dt_mcd)

            # Accumulate log B - log F
            acc += log_b - log_f
            return (q_new, acc, key), None

        # Run K steps using the compiled loop
        (q_K, acc_K, _), _ = jax.lax.scan(step, (q0, 0.0, key), jnp.arange(K_mcd))

        # Add boundary terms: log gamma_1(q_K) - log gamma_0(q_0)
        # This accounts for the "importance weight" at the endpoints
        return log_g1(q_K) - log_g0(q0) + acc_K
        # At optimum: this = -Delta_F, so loss = Delta_F

    def loss_fn(seeds, params):
        """
        Compute the training loss = -mean(ELBO) over a batch of trajectories.

        This function is differentiated with respect to `params` (argnums=1)
        by jax.grad to get the gradient for parameter updates.

        Parameters
        ----------
        seeds  : array of shape (batch_size,) — random seeds for trajectories
        params : network parameters (the variable being optimised)

        Returns
        -------
        loss  : scalar — the loss value (to be minimised)
        (loss,) : auxiliary output (same value, needed for has_aux=True)
        """
        def one(seed):
            k = jax.random.PRNGKey(seed)
            q0, k = _sample_q0(cfg, d, k)
            return elbo_single(q0, params, k)

        # Evaluate elbo_single for each seed in parallel (jax.vmap)
        elbos = jax.vmap(one)(seeds)         # shape (batch_size,)
        loss  = -elbos.mean()                # minimise -ELBO = maximise ELBO
        return loss, (loss,)

    # --- Learning rate schedule: warmup then cosine decay ---
    # The learning rate starts at lr_init, increases to lr_peak during warmup,
    # then slowly decreases to lr_end following a cosine curve.
    # This helps avoid instability at the start and fine-tunes at the end.
    lr_sched = optax.warmup_cosine_decay_schedule(
        init_value=lr_init, peak_value=lr_peak,
        warmup_steps=int(n_epochs * 0.2),
        decay_steps=n_epochs, end_value=lr_end,
    )
    optimizer = optax.adam(lr_sched)
    # Adam is an adaptive learning rate optimizer that adjusts the step size
    # for each parameter individually based on the history of gradients.

    opt_state = optimizer.init(params)

    # jax.grad(loss_fn, argnums=1): differentiate loss_fn with respect to
    # its second argument (params, index 1).
    # has_aux=True: loss_fn returns (loss, aux) and jax.grad returns (grads, aux).
    grad_fn = jax.jit(jax.grad(loss_fn, argnums=1, has_aux=True))
    # jax.jit compiles the whole gradient computation for speed.

    loss_history, best_loss, patience_ctr = [], float("inf"), 0
    key = jax.random.PRNGKey(cfg.seed + 1)

    pbar = tqdm(range(n_epochs), desc="MCD training")
    for epoch in pbar:
        # Sample fresh random seeds for this epoch's batch
        key, sk = jax.random.split(key)
        seeds   = jax.random.randint(sk, (batch_size,), 1, int(1e6))

        # Forward pass: compute loss + gradients in one call
        grads, (loss,) = grad_fn(seeds, params)
        loss_val = float(loss)
        loss_history.append(loss_val)

        # Parameter update: apply the gradient step
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        # Early stopping: stop if loss has not improved for patience epochs
        if loss_val < best_loss:
            best_loss, patience_ctr = loss_val, 0
        else:
            patience_ctr += 1
        if patience_ctr >= early_stop_patience:
            pbar.set_description(f"MCD [early stop @ {epoch+1}]")
            break

    return params, loss_history, meta


# =============================================================================
# Inference (evaluation)
# =============================================================================

def estimate(
    cfg:            PipelineConfig,
    params:         dict,
    meta:           dict,
    emb_dim:        int = 20,
    eval_batch_size: int = 256,
) -> dict:
    """
    Estimate Z1/Z0 using the trained MCD score network.

    Uses the trained backward kernel to compute importance weights
    for N independent trajectories.

    The work formula (eq. 5.18):
        W = log gamma_0(q_0) - log gamma_1(q_K)
            - sum_k [log B^theta_k(q_k | q_{k+1}) - log F_{k+1}(q_{k+1} | q_k)]

    Estimator: mean(exp(-W_i)) -> Z1/Z0

    EMBEDDING INDEX MAPPING:
    The network was trained with K_mcd steps and embedding table of size K_mcd.
    At evaluation, we use K_eval steps (potentially different).
    We map eval step k_eval to training step k_train via:
        k_train = floor(k_eval * dt_eval / dt_train)
    This "resamples" the embedding table to the finer evaluation grid.

    Parameters
    ----------
    cfg    : PipelineConfig — evaluation configuration (may differ from training)
    params : dict — trained network parameters from train()
    meta   : dict — {"K": K_mcd, "dt": dt_mcd} from train()
    emb_dim : int — must match the value used during training
    eval_batch_size : int — mini-batch size for memory management

    Returns
    -------
    dict with keys:
        "works" : np.ndarray of shape (n_samples,)
        "trajs" : None (not stored to save memory)
        "name"  : "MCD"
    """
    d        = cfg.dim
    K        = cfg.n_steps     # number of evaluation steps
    dt       = cfg.dt          # evaluation time step
    sched    = jnp.array(cfg.schedule)
    K_mcd    = meta["K"]       # training steps
    dt_mcd   = meta["dt"]      # training time step
    dt_ratio = dt / dt_mcd     # ratio for embedding index mapping

    V, grad_V, _, log_g0, log_g1 = make_jax_potential(cfg.gmm0, cfg.gmm1)
    _, apply_fn = build_score_network(d, K_mcd, emb_dim)

    def compute_work_single(seed_val):
        """
        Simulate one trajectory and compute its virtual work W.
        This is the evaluation version — no gradient computation needed.
        """
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

            # Map evaluation step to training embedding index
            k_emb  = jnp.clip(jnp.floor(k * dt_ratio).astype(int), 0, K_mcd - 1)
            score  = apply_fn(params, q_new, k_emb)
            mean_b = q_new + dt * grad_V(q_new, lam_kp1) + 2.0 * dt * score
            diff_b = q - mean_b
            log_b  = -(diff_b ** 2).sum() / (4.0 * dt)

            acc += log_b - log_f
            return (q_new, acc, key), None   # no trajectory storage (saves memory)

        (q_K, acc_K, _), _ = jax.lax.scan(step, (q0, 0.0, key), jnp.arange(K))

        # W = -log(IW) where IW is the importance weight
        W = log_g0(q0) - log_g1(q_K) - acc_K
        return W

    works = batched_estimate(
        compute_work_single,
        n_samples=cfg.n_samples,
        seed=cfg.seed + 99,   # different seed from training to avoid correlation
        eval_batch_size=eval_batch_size,
        desc="MCD eval",
    )
    return {"works": works, "trajs": None, "name": "MCD"}
