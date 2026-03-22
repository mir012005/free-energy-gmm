"""
gmm_physics.py
==============
The "physics engine" of the simulation — energy functions for the
interpolated path between pi_0 and pi_1.

PHYSICAL CONTEXT
----------------
We want to reversibly switch a system from one thermodynamic state (pi_0)
to another (pi_1). To do this, we introduce a family of intermediate
distributions parametrised by a "switching parameter" lambda in [0, 1]:

    pi_lambda(q)  proportional to  pi_0(q)^{1-lambda} * pi_1(q)^lambda

At lambda=0  : pi_lambda = pi_0  (the starting state)
At lambda=1  : pi_lambda = pi_1  (the ending state)
In between   : a smooth interpolation mixing both states

This is the standard "thermodynamic integration" or "alchemical switching"
path used in computational chemistry for free energy calculations.

POTENTIAL ENERGY
----------------
In statistical mechanics, a probability distribution pi(q) is related to
a potential energy V(q) by the Boltzmann formula:

    pi(q) = exp(-V(q)) / Z

where Z = integral of exp(-V(q)) dq is the partition function.

Inverting this:  V(q) = -log pi(q)  (up to a constant)

So the potential energy of the interpolated path is:

    V_lambda(q) = -log pi_lambda(q)
                = -(1-lambda) * log gamma_0(q) - lambda * log gamma_1(q)

where gamma_k(q) = unnormalised version of pi_k(q) (see gmm_jax_physics.py
for why we use the unnormalised version).

LANGEVIN DYNAMICS
-----------------
A particle under this potential energy evolves according to the overdamped
Langevin equation (Brownian motion in a potential):

    dq/dt = -grad V_lambda(q)  +  sqrt(2) * noise

The first term is the "drift" — the particle is pushed towards lower
potential energy (higher probability).
The second term is "thermal noise" — random kicks from the environment.

Discretised with time step dt (Euler-Maruyama scheme):

    q_{k+1} = q_k - dt * grad V_lambda(q_k)  +  sqrt(2*dt) * G_k

where G_k ~ N(0, I) is a standard Gaussian random vector.

NOTE ON THIS FILE vs gmm_jax_physics.py
----------------------------------------
This file uses NumPy (Python arrays, CPU only). It is used only by AIS,
which does not need automatic differentiation. The other algorithms (MCD,
CMCD, LED) use gmm_jax_physics.py which uses JAX, a library that can
automatically compute gradients and run on GPU.
"""

from __future__ import annotations
import numpy as np
import numpy.typing as npt
from gmm_config import GMMParams


# =============================================================================
# Core energy functions — operate on a single point q in R^d
# =============================================================================

def _log_gmm(q: npt.NDArray, gmm: GMMParams) -> float:
    """
    Compute the log-density of a single GMM at point q.

    This is just a thin wrapper around gmm.log_density(q) defined in
    gmm_config.py. It returns the NORMALISED log-density:

        log gamma(q) = log [ sum_k  w_k * N(q ; mu_k , Sigma_k) ]

    Parameters
    ----------
    q   : array of shape (d,) — the point to evaluate
    gmm : GMMParams — the distribution to evaluate at q

    Returns
    -------
    float — the log-probability log pi(q)
    """
    return gmm.log_density(q)


def potential_energy(q: npt.NDArray, lam: float,
                     gmm0: GMMParams, gmm1: GMMParams) -> float:
    """
    Compute the potential energy V_lambda(q) of the interpolated path.

    DEFINITION:
        V_lambda(q) = -[(1-lambda) * log gamma_0(q)  +  lambda * log gamma_1(q)]

    This is the NEGATIVE log of the (unnormalised) interpolated density,
    so that the Langevin drift  -grad V_lambda  points towards regions of
    high probability.

    PHYSICAL MEANING:
    At lambda=0, V(q) = -log gamma_0(q), so the particle feels the potential
    of the starting state.
    At lambda=1, V(q) = -log gamma_1(q), so the particle feels the potential
    of the ending state.
    In between, it feels a linear mixture of both.

    Parameters
    ----------
    q    : array of shape (d,)
    lam  : float in [0, 1] — the switching parameter
    gmm0 : GMMParams — the starting distribution
    gmm1 : GMMParams — the ending distribution

    Returns
    -------
    float — the potential energy V_lambda(q)
    """
    return -(1 - lam) * _log_gmm(q, gmm0) - lam * _log_gmm(q, gmm1)


def grad_potential_energy(q: npt.NDArray, lam: float,
                          gmm0: GMMParams, gmm1: GMMParams) -> npt.NDArray:
    """
    Compute the gradient of the potential energy: grad_q V_lambda(q).

    This is the "force" that drives the Langevin particle.
    The particle moves in the direction of  -grad V  (downhill in energy).

    DERIVATION:
        grad V_lambda(q) = -(1-lambda) * grad log gamma_0(q)
                           - lambda    * grad log gamma_1(q)

    This is the linearly interpolated gradient of the two log-densities,
    computed analytically via gmm_config.GMMParams.grad_log_density().

    Parameters
    ----------
    q    : array of shape (d,)
    lam  : float in [0, 1]
    gmm0 : GMMParams
    gmm1 : GMMParams

    Returns
    -------
    array of shape (d,) — the gradient vector at q
    """
    return (-(1 - lam) * gmm0.grad_log_density(q)
            - lam      * gmm1.grad_log_density(q))


def partial_lambda_potential(q: npt.NDArray, lam: float,
                              gmm0: GMMParams, gmm1: GMMParams) -> float:
    """
    Compute the partial derivative of V_lambda with respect to lambda:

        dV_lambda / d_lambda = log gamma_0(q) - log gamma_1(q)

    PHYSICAL MEANING — THE WORK FORMULA:
    This quantity is central to the Jarzynski work calculation. As lambda
    increases by a small amount d_lambda, the work done on the system is:

        dW = (dV_lambda / d_lambda) * d_lambda
           = [log gamma_0(q) - log gamma_1(q)] * d_lambda

    Summing over all steps gives the total work W accumulated along a trajectory.
    By Jarzynski's equality:  E[exp(-W)] = Z1/Z0 = exp(-Delta_F)

    Note that this derivative does NOT depend on lambda — it is simply the
    log-density ratio between the two endpoint distributions.

    Parameters
    ----------
    q    : array of shape (d,)
    lam  : float (not actually used, kept for API consistency)
    gmm0 : GMMParams
    gmm1 : GMMParams

    Returns
    -------
    float — the lambda-derivative of the potential energy
    """
    return _log_gmm(q, gmm0) - _log_gmm(q, gmm1)


# =============================================================================
# Vectorised wrappers — apply the above functions to batches of n particles
# =============================================================================

def batch_potential_energy(qs: npt.NDArray, lam: float,
                            gmm0: GMMParams, gmm1: GMMParams) -> npt.NDArray:
    """
    Compute V_lambda(q_i) for each of n particles.

    Parameters
    ----------
    qs : array of shape (n, d) — n particle positions

    Returns
    -------
    array of shape (n,) — one energy value per particle
    """
    return np.array([potential_energy(qs[i], lam, gmm0, gmm1)
                     for i in range(qs.shape[0])])


def batch_grad_potential(qs: npt.NDArray, lam: float,
                          gmm0: GMMParams, gmm1: GMMParams) -> npt.NDArray:
    """
    Compute grad V_lambda(q_i) for each of n particles.

    Parameters
    ----------
    qs : array of shape (n, d)

    Returns
    -------
    array of shape (n, d) — one gradient vector per particle
    """
    return np.stack([grad_potential_energy(qs[i], lam, gmm0, gmm1)
                     for i in range(qs.shape[0])])


def batch_partial_lambda(qs: npt.NDArray, lam: float,
                          gmm0: GMMParams, gmm1: GMMParams) -> npt.NDArray:
    """
    Compute dV/d_lambda(q_i) for each of n particles.

    Parameters
    ----------
    qs : array of shape (n, d)

    Returns
    -------
    array of shape (n,) — one work contribution per particle
    """
    return np.array([partial_lambda_potential(qs[i], lam, gmm0, gmm1)
                     for i in range(qs.shape[0])])


def sample_initial(gmm0: GMMParams, n: int,
                   rng: np.random.Generator | None = None) -> npt.NDArray:
    """
    Sample n particle positions from the starting distribution pi_0.

    These are the initial conditions q_0 for the Langevin trajectories.
    Each trajectory starts from an independent sample from pi_0.

    Parameters
    ----------
    gmm0 : GMMParams — the starting distribution
    n    : int — number of particles (trajectories)
    rng  : optional numpy random generator for reproducibility

    Returns
    -------
    array of shape (n, d)
    """
    return gmm0.sample(n, rng=rng)
