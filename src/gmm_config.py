"""
gmm_config.py
─────────────
Dataclasses that describe a Gaussian Mixture Model distribution
and the runtime configuration for the comparison pipeline.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import numpy.typing as npt


# ─────────────────────────────────────────────────────────────────────────────
# GMM distribution descriptor
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GMMParams:
    """
    Describes a Gaussian Mixture Model in R^d.

    Attributes
    ----------
    means   : (K, d) array  – one mean vector per component
    covs    : (K, d, d) array – one covariance matrix per component
    weights : (K,) array     – mixture weights (must sum to 1)
    """
    means: npt.NDArray    # shape (K, d)
    covs: npt.NDArray     # shape (K, d, d)
    weights: npt.NDArray  # shape (K,)

    def __post_init__(self):
        self.means   = np.asarray(self.means,   dtype=float)
        self.covs    = np.asarray(self.covs,    dtype=float)
        self.weights = np.asarray(self.weights, dtype=float)

        K_m, d      = self.means.shape
        K_c, d1, d2 = self.covs.shape
        K_w,        = self.weights.shape

        assert K_m == K_c == K_w,  "means, covs and weights must have the same K"
        assert d == d1 == d2,      "covariance matrices must be (d, d)"
        assert np.isclose(self.weights.sum(), 1.0), "weights must sum to 1"

    @property
    def n_components(self) -> int:
        return self.means.shape[0]

    @property
    def dim(self) -> int:
        return self.means.shape[1]

    # ── helpers ──────────────────────────────────────────────────────────────

    def log_density(self, q: npt.NDArray) -> float:
        """Unnormalized log density  log γ(q) = log Σ_k w_k N(q; μ_k, Σ_k)."""
        from scipy.stats import multivariate_normal
        log_probs = np.array([
            np.log(self.weights[k] + 1e-300)
            + multivariate_normal.logpdf(q, mean=self.means[k], cov=self.covs[k])
            for k in range(self.n_components)
        ])
        return float(np.logaddexp.reduce(log_probs))

    def grad_log_density(self, q: npt.NDArray) -> npt.NDArray:
        """
        ∇_q log p(q)  computed via the identity
            ∇ log p = Σ_k r_k(q) · (-Σ_k^{-1} (q - μ_k))
        where r_k(q) are the posterior responsibilities.
        """
        from scipy.stats import multivariate_normal
        log_probs = np.array([
            np.log(self.weights[k] + 1e-300)
            + multivariate_normal.logpdf(q, mean=self.means[k], cov=self.covs[k])
            for k in range(self.n_components)
        ])
        # numerically stable responsibilities
        log_probs -= np.logaddexp.reduce(log_probs)
        resp = np.exp(log_probs)                          # (K,)

        grad = np.zeros_like(q)
        for k in range(self.n_components):
            prec = np.linalg.inv(self.covs[k])
            grad += resp[k] * (-prec @ (q - self.means[k]))
        return grad

    def sample(self, n: int, rng: np.random.Generator | None = None) -> npt.NDArray:
        """Draw n i.i.d. samples from the GMM."""
        if rng is None:
            rng = np.random.default_rng()
        indices = rng.choice(self.n_components, size=n, p=self.weights)
        samples = np.zeros((n, self.dim))
        for k in range(self.n_components):
            mask = indices == k
            nk = mask.sum()
            if nk > 0:
                samples[mask] = rng.multivariate_normal(
                    self.means[k], self.covs[k], size=nk
                )
        return samples                                    # (n, d)

    # ── convenience constructors ─────────────────────────────────────────────

    @classmethod
    def single_gaussian(cls, mean: npt.NDArray, cov: npt.NDArray) -> "GMMParams":
        """Wrap a single Gaussian as a 1-component GMM."""
        mean = np.atleast_1d(mean)
        cov  = np.atleast_2d(cov)
        if cov.ndim == 1:                         # diagonal supplied as vector
            cov = np.diag(cov)
        return cls(
            means=mean[None, :],
            covs=cov[None, :, :],
            weights=np.array([1.0]),
        )

    @classmethod
    def isotropic(cls, means: npt.NDArray, var: float,
                  weights: npt.NDArray | None = None) -> "GMMParams":
        """All components share the same isotropic covariance σ²I."""
        means = np.atleast_2d(means)
        K, d  = means.shape
        covs  = np.stack([var * np.eye(d)] * K)
        if weights is None:
            weights = np.ones(K) / K
        return cls(means=means, covs=covs, weights=weights)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    """
    Runtime parameters shared by all algorithms.

    Attributes
    ----------
    gmm0        : start distribution  (π_0)
    gmm1        : end   distribution  (π_1)
    T           : total simulation time
    dt          : time step
    n_samples   : number of independent trajectories
    schedule    : λ_k values; if None, linear schedule is built automatically
    seed        : master RNG seed
    algorithms  : list of algorithm names to run
                  choices: "ais", "mcd_no_nn", "escorted", "mcd_nn"
    """
    gmm0: GMMParams
    gmm1: GMMParams
    T: float            = 1.0
    dt: float           = 1e-3
    n_samples: int      = 1000
    schedule: npt.NDArray | None = None
    seed: int           = 42
    algorithms: list[str] = field(
        default_factory=lambda: ["ais", "mcd_no_nn", "escorted"]
    )
    # MCD-NN specific
    n_epochs: int        = 500
    batch_size: int      = 128
    mcd_nn_params: dict | None = None   # pre-trained params; train if None

    def __post_init__(self):
        assert self.gmm0.dim == self.gmm1.dim, \
            "π_0 and π_1 must live in the same dimension"
        if self.schedule is None:
            n_steps = int(self.T / self.dt)
            self.schedule = np.linspace(0.0, 1.0, n_steps + 1)

    @property
    def dim(self) -> int:
        return self.gmm0.dim

    @property
    def n_steps(self) -> int:
        return len(self.schedule) - 1
