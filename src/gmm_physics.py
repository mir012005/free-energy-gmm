"""
gmm_physics.py
──────────────
Energy functions for the interpolated GMM path:

    π_λ(q) ∝ π_0(q)^{1-λ} · π_1(q)^λ

The un-normalised log density is

    log γ_λ(q) = (1-λ) log γ_0(q) + λ log γ_1(q)

where γ_k(q) = Σ_j w_j^(k) N(q; μ_j^(k), Σ_j^(k)).

All functions here operate on a single configuration q ∈ R^d (1-D NumPy arrays).
Vectorisation over a batch of particles is handled at the algorithm level.
"""

from __future__ import annotations
import numpy as np
import numpy.typing as npt
from gmm_config import GMMParams


# ─────────────────────────────────────────────────────────────────────────────
# Core energy functions
# ─────────────────────────────────────────────────────────────────────────────

def _log_gmm(q: npt.NDArray, gmm: GMMParams) -> float:
    """log γ(q) for a single GMM (un-normalised)."""
    return gmm.log_density(q)


def potential_energy(q: npt.NDArray, lam: float,
                     gmm0: GMMParams, gmm1: GMMParams) -> float:
    """
    V_λ(q) = -(log γ_λ(q))
           = -[(1-λ) log γ_0(q) + λ log γ_1(q)]

    This is the *negative* log un-normalised density so that the overdamped
    Langevin drift is  -∇V_λ = ∇ log γ_λ.
    """
    return -(1 - lam) * _log_gmm(q, gmm0) - lam * _log_gmm(q, gmm1)


def grad_potential_energy(q: npt.NDArray, lam: float,
                          gmm0: GMMParams, gmm1: GMMParams) -> npt.NDArray:
    """
    ∇_q V_λ(q)  (shape: (d,))
    """
    return -(1 - lam) * gmm0.grad_log_density(q) - lam * gmm1.grad_log_density(q)


def partial_lambda_potential(q: npt.NDArray, lam: float,
                             gmm0: GMMParams, gmm1: GMMParams) -> float:
    """
    ∂V_λ / ∂λ = log γ_0(q) - log γ_1(q)
    (used in the standard Jarzynski work accumulation)
    """
    return _log_gmm(q, gmm0) - _log_gmm(q, gmm1)


# ─────────────────────────────────────────────────────────────────────────────
# Vectorised wrappers  (batch of n particles  →  shape (n,) or (n, d))
# ─────────────────────────────────────────────────────────────────────────────

def batch_potential_energy(qs: npt.NDArray, lam: float,
                            gmm0: GMMParams, gmm1: GMMParams) -> npt.NDArray:
    """qs: (n, d) → (n,)"""
    return np.array([potential_energy(qs[i], lam, gmm0, gmm1)
                     for i in range(qs.shape[0])])


def batch_grad_potential(qs: npt.NDArray, lam: float,
                          gmm0: GMMParams, gmm1: GMMParams) -> npt.NDArray:
    """qs: (n, d) → (n, d)"""
    return np.stack([grad_potential_energy(qs[i], lam, gmm0, gmm1)
                     for i in range(qs.shape[0])])


def batch_partial_lambda(qs: npt.NDArray, lam: float,
                          gmm0: GMMParams, gmm1: GMMParams) -> npt.NDArray:
    """qs: (n, d) → (n,)"""
    return np.array([partial_lambda_potential(qs[i], lam, gmm0, gmm1)
                     for i in range(qs.shape[0])])


# ─────────────────────────────────────────────────────────────────────────────
# Sampling from π_{Δt,0}  (initial distribution adjusted for EM discretisation)
# ─────────────────────────────────────────────────────────────────────────────

def sample_initial(gmm0: GMMParams, n: int,
                   rng: np.random.Generator | None = None) -> npt.NDArray:
    """
    Sample n particles from π_0 (the starting GMM).
    For the purposes of this pipeline we sample exactly from π_0; the
    O(Δt) correction to the invariant measure of the Euler–Maruyama kernel
    is negligible for small Δt.
    Returns shape (n, d).
    """
    return gmm0.sample(n, rng=rng)
