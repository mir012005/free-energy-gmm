"""
gmm_config.py
=============
Data structures that describe the problem and the simulation parameters.

PHYSICAL CONTEXT
----------------
We want to estimate the free energy difference between two thermodynamic
states described by Gaussian Mixture Model (GMM) probability distributions:

    pi_0(q)  =  start state  (e.g. a particle not interacting with the system)
    pi_1(q)  =  end state    (e.g. the same particle fully inserted)

q is the position of the particle(s), living in R^d.

A GMM is a weighted sum of Gaussian distributions:

    pi(q) = sum_{k=1}^{K}  w_k * N(q ; mu_k , Sigma_k)

where:
    w_k     = weight of component k  (all weights sum to 1)
    mu_k    = mean (centre) of component k  — a d-dimensional vector
    Sigma_k = covariance matrix of component k  — a d x d matrix

For a single Gaussian (K=1) this reduces to the standard normal distribution.
For multiple Gaussians (K>1) this can represent multimodal distributions,
i.e. distributions with several "peaks" — useful for modelling systems with
multiple stable states.

WHAT THIS FILE PROVIDES
-----------------------
Two Python dataclasses:

    GMMParams      — describes a single GMM distribution (one of pi_0 or pi_1)
    PipelineConfig — groups all simulation parameters together in one place

A dataclass is like a structured container: it stores data and validates it
automatically when you create it. Think of it as a labelled box that checks
its own contents.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import numpy.typing as npt


# =============================================================================
# GMMParams — description of one GMM distribution
# =============================================================================

@dataclass
class GMMParams:
    """
    Describes a Gaussian Mixture Model (GMM) in R^d.

    A GMM is a probability distribution of the form:
        pi(q) = sum_{k=1}^{K}  w_k * N(q ; mu_k , Sigma_k)

    This class stores the three ingredients of a GMM:
        means   : the K centre positions (one per component)
        covs    : the K covariance matrices (shape of each Gaussian)
        weights : the K mixture weights (how much each component contributes)

    Parameters
    ----------
    means   : array of shape (K, d)
              Each row is the mean vector of one component.
              Example for 2 components in 2D:
                  means = [[-2.0, 0.0],   <- centre of component 1
                           [ 2.0, 0.0]]   <- centre of component 2

    covs    : array of shape (K, d, d)
              Each slice covs[k] is the covariance matrix of component k.
              A diagonal matrix diag(s1^2, s2^2) means the Gaussian has
              standard deviations s1 along axis 1 and s2 along axis 2,
              with no correlation between axes.
              Example for isotropic (spherical) Gaussians with variance 1:
                  covs = [[[1.0, 0.0],    <- covariance of component 1
                           [0.0, 1.0]],
                          [[1.0, 0.0],    <- covariance of component 2
                           [0.0, 1.0]]]

    weights : array of shape (K,)
              The weight w_k of each component. Must sum to 1.
              Example: weights = [0.3, 0.7]  means 30% component 1, 70% component 2.

    Raises
    ------
    AssertionError if the shapes are inconsistent or weights do not sum to 1.
    """

    means:   npt.NDArray   # shape (K, d)
    covs:    npt.NDArray   # shape (K, d, d)
    weights: npt.NDArray   # shape (K,)

    def __post_init__(self):
        """
        Called automatically after __init__.
        Converts inputs to numpy arrays and checks that shapes are consistent.
        """
        # Convert to float numpy arrays — handles lists, tuples, etc.
        self.means   = np.asarray(self.means,   dtype=float)
        self.covs    = np.asarray(self.covs,    dtype=float)
        self.weights = np.asarray(self.weights, dtype=float)

        # Unpack shapes for validation
        K_m, d      = self.means.shape           # K components, d dimensions
        K_c, d1, d2 = self.covs.shape            # K covariance matrices of size d x d
        K_w,        = self.weights.shape         # K weights

        # All three must have the same number of components K
        assert K_m == K_c == K_w, (
            f"means has {K_m} components, covs has {K_c}, weights has {K_w}. "
            f"They must all be equal."
        )
        # Covariance matrices must be square and match the dimension d
        assert d == d1 == d2, (
            f"dimension mismatch: means says d={d} but covs says d={d1}x{d2}"
        )
        # Weights must be a valid probability distribution
        assert np.isclose(self.weights.sum(), 1.0), (
            f"weights sum to {self.weights.sum():.6f}, must sum to 1.0"
        )

    # -------------------------------------------------------------------------
    # Properties — convenient ways to access derived information
    # -------------------------------------------------------------------------

    @property
    def n_components(self) -> int:
        """Number of Gaussian components K in the mixture."""
        return self.means.shape[0]

    @property
    def dim(self) -> int:
        """Dimension d of the space (number of coordinates per point)."""
        return self.means.shape[1]

    # -------------------------------------------------------------------------
    # log_density — evaluate the GMM log-probability at a point q
    # -------------------------------------------------------------------------

    def log_density(self, q: npt.NDArray) -> float:
        """
        Compute the log-probability of a point q under this GMM:

            log pi(q) = log [ sum_k  w_k * N(q ; mu_k , Sigma_k) ]

        The log is used for numerical stability — probabilities can be very
        small numbers and taking their log keeps them in a tractable range.

        This uses scipy's multivariate_normal which returns the NORMALISED
        log-density, i.e. it includes the normalisation constant 1/sqrt(|2*pi*Sigma|).

        Parameters
        ----------
        q : array of shape (d,)
            The point at which to evaluate the density.

        Returns
        -------
        float
            The log-probability log pi(q).

        Implementation note
        -------------------
        We compute log p_k = log w_k + log N(q ; mu_k , Sigma_k) for each
        component k, then combine them using the log-sum-exp trick:
            log(a + b) = log(exp(log_a) + exp(log_b))
        This avoids numerical underflow when individual probabilities are tiny.
        """
        from scipy.stats import multivariate_normal

        # Compute log-probability under each component separately
        log_probs = np.array([
            np.log(self.weights[k] + 1e-300)   # 1e-300 avoids log(0)
            + multivariate_normal.logpdf(q, mean=self.means[k], cov=self.covs[k])
            for k in range(self.n_components)
        ])
        # Combine with numerically stable log-sum-exp
        return float(np.logaddexp.reduce(log_probs))

    # -------------------------------------------------------------------------
    # grad_log_density — gradient of the log-probability (the "force")
    # -------------------------------------------------------------------------

    def grad_log_density(self, q: npt.NDArray) -> npt.NDArray:
        """
        Compute the gradient of the log-density with respect to q:

            grad_q  log pi(q)

        PHYSICAL MEANING: In Langevin dynamics, the particle is pushed by
        the force  -grad V = +grad log pi. This gradient therefore tells
        the particle which direction to move to increase its probability,
        i.e. it points towards the nearest density peak.

        MATHEMATICAL DERIVATION:
        Using the identity  grad log pi = (1/pi) * grad pi  and the
        mixture structure, one can show:

            grad log pi(q) = sum_k  r_k(q) * grad log N(q ; mu_k , Sigma_k)
                           = sum_k  r_k(q) * (- Sigma_k^{-1} (q - mu_k))

        where r_k(q) = w_k * N(q ; mu_k , Sigma_k) / pi(q)  are the
        "responsibilities" — the posterior probability that point q was
        generated by component k.

        For a single Gaussian this reduces to the familiar result:
            grad log N(q ; mu , Sigma) = -Sigma^{-1} (q - mu)

        Parameters
        ----------
        q : array of shape (d,)

        Returns
        -------
        array of shape (d,)
            The gradient vector at q.
        """
        from scipy.stats import multivariate_normal

        # Step 1: compute log-probability under each component
        log_probs = np.array([
            np.log(self.weights[k] + 1e-300)
            + multivariate_normal.logpdf(q, mean=self.means[k], cov=self.covs[k])
            for k in range(self.n_components)
        ])

        # Step 2: convert to responsibilities r_k(q) in a numerically stable way
        # Subtract the maximum before taking exp to avoid overflow
        log_probs -= np.logaddexp.reduce(log_probs)   # normalise in log-space
        resp = np.exp(log_probs)                       # shape (K,), sums to 1

        # Step 3: compute gradient as responsibility-weighted sum of per-component gradients
        grad = np.zeros_like(q)
        for k in range(self.n_components):
            prec = np.linalg.inv(self.covs[k])              # precision matrix Sigma_k^{-1}
            grad += resp[k] * (-prec @ (q - self.means[k])) # -Sigma^{-1}(q - mu_k)
        return grad

    # -------------------------------------------------------------------------
    # sample — draw random points from the GMM
    # -------------------------------------------------------------------------

    def sample(self, n: int, rng: np.random.Generator | None = None) -> npt.NDArray:
        """
        Draw n independent random samples from this GMM.

        ALGORITHM (ancestral sampling):
            1. For each sample, randomly pick a component k with probability w_k.
            2. Draw a point from the Gaussian N(mu_k, Sigma_k) of that component.

        This produces samples that follow the mixture distribution exactly.

        Parameters
        ----------
        n   : number of samples to draw
        rng : optional numpy random generator for reproducibility.
              If None, a fresh generator is created (non-reproducible).

        Returns
        -------
        array of shape (n, d)
            Each row is one sample.
        """
        if rng is None:
            rng = np.random.default_rng()

        # Step 1: assign each sample to a component
        indices = rng.choice(self.n_components, size=n, p=self.weights)

        # Step 2: draw from the corresponding Gaussian
        samples = np.zeros((n, self.dim))
        for k in range(self.n_components):
            mask = indices == k      # which samples belong to component k
            nk   = mask.sum()        # how many samples for this component
            if nk > 0:
                samples[mask] = rng.multivariate_normal(
                    self.means[k], self.covs[k], size=nk)

        return samples   # shape (n, d)

    # -------------------------------------------------------------------------
    # Convenience constructors — quick ways to create common GMMs
    # -------------------------------------------------------------------------

    @classmethod
    def single_gaussian(cls, mean: npt.NDArray, cov: npt.NDArray) -> "GMMParams":
        """
        Create a GMM with a single Gaussian component (K=1).

        This is the simplest case — a single Gaussian N(mean, cov).

        Parameters
        ----------
        mean : array of shape (d,)    — the centre of the Gaussian
        cov  : array of shape (d, d)  — the covariance matrix
               Can also pass a 1D array of shape (d,) which will be
               interpreted as the diagonal of the covariance matrix.

        Example
        -------
            # 1D Gaussian N(-2, 1):
            gmm = GMMParams.single_gaussian(mean=[-2.0], cov=[[1.0]])

            # 2D Gaussian N([0,0], I):
            gmm = GMMParams.single_gaussian(mean=[0., 0.], cov=np.eye(2))
        """
        mean = np.atleast_1d(mean)
        cov  = np.atleast_2d(cov)
        if cov.ndim == 1:          # diagonal supplied as a vector
            cov = np.diag(cov)
        return cls(
            means=mean[None, :],   # add K dimension: shape (1, d)
            covs=cov[None, :, :],  # add K dimension: shape (1, d, d)
            weights=np.array([1.0]),
        )

    @classmethod
    def isotropic(
        cls,
        means: npt.NDArray,
        var: float,
        weights: npt.NDArray | None = None,
    ) -> "GMMParams":
        """
        Create a GMM where all components share the same isotropic covariance.

        "Isotropic" means the Gaussian is spherical (same variance in all
        directions, no correlations). The covariance matrix is sigma^2 * I
        where I is the identity matrix.

        Parameters
        ----------
        means   : array of shape (K, d) — one mean per component
        var     : float — the common variance sigma^2 for all components
        weights : array of shape (K,) — optional, defaults to uniform (1/K each)

        Example
        -------
            # Two components in 2D, each with variance 1:
            gmm = GMMParams.isotropic(
                means=[[-3., 0.], [3., 0.]],
                var=1.0
            )
        """
        means = np.atleast_2d(means)
        K, d  = means.shape
        # Build K identical covariance matrices: var * I_d
        covs  = np.stack([var * np.eye(d)] * K)
        if weights is None:
            weights = np.ones(K) / K   # uniform weights
        return cls(means=means, covs=covs, weights=weights)


# =============================================================================
# PipelineConfig — all simulation parameters in one place
# =============================================================================

@dataclass
class PipelineConfig:
    """
    Groups all parameters needed to run a free energy simulation.

    This is the single object passed to every algorithm. Instead of passing
    many separate arguments, you create one PipelineConfig and pass it around.

    SIMULATION PARAMETERS
    ---------------------
    The overdamped Langevin dynamics is discretised using the Euler-Maruyama
    scheme with time step dt over a total time T, giving K = T/dt steps.
    The particle position evolves as:

        q_{k+1} = q_k  -  dt * grad V_{lambda_{k+1}}(q_k)  +  sqrt(2*dt) * noise

    where V_lambda is the interpolated potential and lambda goes from 0 to 1.

    Parameters
    ----------
    gmm0        : GMMParams
                  The start distribution pi_0 (sampled at t=0).

    gmm1        : GMMParams
                  The end distribution pi_1 (target at t=T).

    T           : float, default 1.0
                  Total simulation time. Larger T means a slower, more
                  reversible switch between pi_0 and pi_1, which reduces
                  the variance of the free energy estimator.

    dt          : float, default 1e-3
                  Time step for the Euler-Maruyama discretisation.
                  Smaller dt = more accurate but slower.

    n_samples   : int, default 1000
                  Number of independent trajectories to simulate.
                  More samples = better statistical estimate.

    schedule    : array of shape (K+1,), optional
                  The sequence lambda_0=0, lambda_1, ..., lambda_K=1
                  controlling how fast the system switches from pi_0 to pi_1.
                  Default: linear schedule lambda_k = k/K.
                  Non-linear schedules can reduce variance (e.g. power law).

    seed        : int, default 42
                  Random seed for reproducibility. Using the same seed
                  gives identical results across runs.
    """

    gmm0:     GMMParams
    gmm1:     GMMParams
    T:        float           = 1.0
    dt:       float           = 1e-3
    n_samples: int            = 1000
    schedule: npt.NDArray | None = None
    seed:     int             = 42
    # These fields are stored but mainly used by run_experiment.py
    algorithms:  list[str]   = field(default_factory=lambda: ["ais", "mcd", "cmcd", "led"])
    n_epochs:    int         = 500
    batch_size:  int         = 128

    def __post_init__(self):
        """Validate inputs and build the schedule if not provided."""
        assert self.gmm0.dim == self.gmm1.dim, (
            f"pi_0 has dimension {self.gmm0.dim} but pi_1 has dimension {self.gmm1.dim}. "
            f"Both distributions must live in the same space."
        )
        if self.schedule is None:
            # Default: linear interpolation lambda_k = k/K
            n_steps = int(self.T / self.dt)
            self.schedule = np.linspace(0.0, 1.0, n_steps + 1)

    @property
    def dim(self) -> int:
        """Dimension d of the configuration space."""
        return self.gmm0.dim

    @property
    def n_steps(self) -> int:
        """
        Number of integration steps K = len(schedule) - 1.
        The schedule has K+1 values: lambda_0, lambda_1, ..., lambda_K.
        """
        return len(self.schedule) - 1
