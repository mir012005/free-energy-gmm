from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import numpy.typing as npt
import jax
import jax.numpy as jnp

@dataclass
class GMMParams:
    """Modèle de Mélange Gaussien dans R^d."""
    means: npt.NDArray    # (k, d)
    covs: npt.NDArray     # (k, d, d)
    weights: npt.NDArray  # (k,)

    def __post_init__(self):
        self.means   = np.asarray(self.means, dtype=float)
        self.covs    = np.asarray(self.covs, dtype=float)
        self.weights = np.asarray(self.weights, dtype=float)

        assert self.means.shape[0] == self.covs.shape[0] == self.weights.shape[0], "Dimensions k incohérentes"
        assert np.isclose(self.weights.sum(), 1.0), "Les poids doivent sommer à 1"

    @property
    def dim(self) -> int:
        return self.means.shape[1]

    @classmethod
    def single_gaussian(cls, mean: npt.NDArray, cov: npt.NDArray) -> "GMMParams":
        """Crée un GMM à une seule composante."""
        mean = np.atleast_1d(mean)
        cov  = np.atleast_2d(cov)
        if cov.ndim == 1:
            cov = np.diag(cov)
        return cls(means=mean[None, :], covs=cov[None, :, :], weights=np.array([1.0]))

@dataclass
class PipelineConfig:
    """Paramètres d'exécution partagés par tous les algorithmes."""
    gmm0: 'GMMParams' # Assure-toi que GMMParams est bien importé
    gmm1: 'GMMParams'
    T: float = 1.0
    n_steps: int = 1000      # Nombre d'étapes générique
    n_samples: int = 10000  # Nombre de trajectoires
    seed: int = 42

    n_epochs: int = 10000
    batch_size_train: int = 256
    batch_size_val: int = 256
    lr_init: float = 5e-4
    patience: int = 500
    emb_dim: int = 64
    clip_norm: float = 1.0
    weight_decay: float = 0.0

    # On définit juste les valeurs par défaut ici (avec leur type)
    dt_train: float = 1e-3 
    dt_eval: float = 1e-4 

    # On dit à dataclass de préparer ces variables, mais de ne pas les demander à l'utilisateur
    n_steps_train: int = field(init=False)
    n_steps_eval: int = field(init=False)
    schedule_train: np.ndarray = field(init=False)
    schedule_eval:  np.ndarray = field(init=False)

    def __post_init__(self):
        assert self.gmm0.dim == self.gmm1.dim, "π_0 et π_1 doivent avoir la même dimension"

        self.n_steps_train = int(self.T / self.dt_train)
        self.n_steps_eval = int(self.T / self.dt_eval)
        
        self.schedule_train = np.linspace(0.0, 1.0, self.n_steps_train + 1)
        self.schedule_eval  = np.linspace(0.0, 1.0, self.n_steps_eval  + 1)

    @property
    def dim(self) -> int:
        return self.gmm0.dim
    """
    @property
    def dt(self) -> float:
        return self.T / self.n_steps

    @property
    def schedule(self) -> npt.NDArray:
        return np.linspace(0.0, 1.0, self.n_steps + 1)
    """

def make_jax_potential(gmm0: GMMParams, gmm1: GMMParams):
    """
    Génère les fonctions d'énergie JAX compilées.
    log_g0 et log_g1 retournent les densités non normalisées.
    """
    means0, covs0, weights0 = jnp.array(gmm0.means), jnp.array(gmm0.covs), jnp.array(gmm0.weights)
    means1, covs1, weights1 = jnp.array(gmm1.means), jnp.array(gmm1.covs), jnp.array(gmm1.weights)

    def _log_gmm_unnorm(q, means, covs, weights):
        # Astuce LogSumExp : Évite l'underflow numérique (erreurs NaN).
        # Si la particule q est très éloignée du centre des gaussiennes, exp(-distance) s'arrondirait à 0 absolu, faisant planter l'algorithme au moment du calcul de log(0). 
        # logsumexp factorise la valeur maximale pour garantir une stabilité absolue : log(Σ exp(x)) = max(x) + log(Σ exp(x - max(x)))
        def _log_une_seule_component(mean, cov, w):
            diff = q - mean
            prec = jnp.linalg.inv(cov) # matrice de précision qui est l'inverse de la matrice de covariance
            mah_dist  = diff @ prec @ diff # calcule la distance de Mahalanobis : (q−μ)T * Σ^(−1) * (q−μ)
            return jnp.log(w + 1e-300) - 0.5 * mah_dist
        log_probs = jax.vmap(_log_une_seule_component)(means, covs, weights)
        return jax.scipy.special.logsumexp(log_probs)

    def log_gamma0(q): return _log_gmm_unnorm(q, means0, covs0, weights0)
    def log_gamma1(q): return _log_gmm_unnorm(q, means1, covs1, weights1)

    def V(q, lam):
        # V(q) = − ln(γ(q))
        return -(1.0 - lam) * log_gamma0(q) - lam * log_gamma1(q)

    def dV_dlam(q, lam):
        return log_gamma0(q) - log_gamma1(q)

    grad_V = jax.grad(V, argnums=0)

    return (
        jax.jit(V),
        jax.jit(grad_V),
        jax.jit(dV_dlam),
        jax.jit(log_gamma0),
        jax.jit(log_gamma1),
    )