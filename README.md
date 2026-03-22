# Free Energy Estimation via Non-Equilibrium Langevin Dynamics

**Benchmarking ML-based free energy estimators — AIS, MCD, CMCD, LED — on Gaussian Mixture Models**

---

## Overview

This project implements and compares four algorithms for computing **free energy differences** between probability distributions using non-equilibrium Langevin dynamics. It is based on Chapter 5 of the PhD thesis *"Foundations and Optimization of Langevin Samplers"* (Santet, 2024, École des Ponts ParisTech).

The central problem is estimating the ratio:

$$\frac{Z_1}{Z_0} = e^{-\Delta F}$$

where $Z_0, Z_1$ are the partition functions of two Gaussian Mixture Model (GMM) distributions $\pi_0$ and $\pi_1$, connected by the interpolated potential:

$$V_\lambda(q) = -(1-\lambda)\log\gamma_0(q) - \lambda\log\gamma_1(q), \quad \lambda \in [0,1]$$

All algorithms simulate overdamped Langevin dynamics along this path and collect importance weights to estimate the ratio.

---

## Algorithms

| Algorithm | Type | Key idea | Zero-variance? |
|---|---|---|---|
| **AIS** | Baseline | Standard Jarzynski estimator, no learning | No |
| **MCD** | Score matching | Learns score $s_\theta \approx \nabla\log p_k$ via ELBO; reuses fixed trajectory batch | No |
| **CMCD** | Drift learning | Learns escorting drift $u_\theta$ in forward kernel; resamples each epoch | Yes (in theory) |
| **LED** | Drift learning | Minimizes $\text{Var}(W^\theta)$ directly via work formula + Hutchinson divergence | Yes (in theory) |

### Theoretical background

**AIS** accumulates the work $W = \sum_k (V_{\lambda_{k+1}} - V_{\lambda_k})(q_k)$ along standard EM trajectories. By Jarzynski's equality: $\mathbb{E}[\exp(-W)] = Z_1/Z_0$.

**MCD** (Section 5.3.2) approximates the optimal backward kernel via a neural network score, maximising the path-space ELBO. The key advantage: the forward measure $Q$ does not depend on $\theta$, so trajectories are generated once and reused.

**CMCD** (Section 5.3.3) parametrises the *forward* dynamics with a learned drift $u_\theta$, matching a score-free time-reversal kernel. Requires resampling trajectories at every gradient step.

**LED** (Section 5.2.2) directly minimises $\text{Var}(W^\theta)$ (log-variance loss), where the work involves the divergence $\nabla \cdot u_\theta$ estimated via Hutchinson's estimator.

---

## Results — 1D Gaussian baseline (Section 5.4.1)

$\pi_0 = \mathcal{N}(-2, 1)$,  $\pi_1 = \mathcal{N}(2, 0.25)$,  reference $Z_1/Z_0 = 0.5$

| Algorithm | Ratio estimate | $\Delta F$ estimate | Var(ratio) | VR vs AIS |
|---|---|---|---|---|
| AIS | 0.638 | 0.450 | 1.04×10³ | 1× |
| MCD | **0.491** | **0.710** | 8.1×10⁻¹ | **1276×** |
| CMCD | **0.501** | **0.691** | 5.5×10⁻¹ | **1874×** |
| LED | 0.360 | 1.021 | 1.5×10⁻⁵ | 69M× (biased) |

MCD and CMCD closely reproduce the results from Table 1 of the thesis. LED achieves extraordinary variance reduction but with a systematic bias due to the degenerate nature of the $\text{Var}(W)$ objective.

---

## Project structure

```
free-energy-gmm/
│
├── src/                        # Core library
│   ├── gmm_config.py           # GMMParams and PipelineConfig dataclasses
│   ├── gmm_physics.py          # NumPy energy functions (used by AIS)
│   ├── gmm_jax_physics.py      # JAX energy functions (used by MCD/CMCD/LED)
│   ├── gmm_networks.py         # Shared ResNet architecture (score/drift network)
│   ├── gmm_eval_utils.py       # Batched evaluation loop (avoids OOM)
│   ├── gmm_ais.py              # AIS algorithm
│   ├── gmm_mcd.py              # MCD: train() + estimate()
│   ├── gmm_cmcd.py             # CMCD: train() + estimate()
│   ├── gmm_led.py              # LED: train() + estimate()
│   ├── gmm_metrics.py          # Metrics table and variance reduction
│   └── gmm_plots.py            # Work histograms, convergence, loss curves
│
├── experiments/
│   ├── run_experiment.py       # Single experiment (quick test)
│   └── run_overnight.py        # Multi-experiment pipeline with auto-visualisation
│
├── assets/                     # Images used in this README
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Installation

```bash
git clone https://github.com/<your-username>/free-energy-gmm.git
cd free-energy-gmm
pip install -r requirements.txt
```

> **Windows users**: JAX 0.5.x has known DLL issues on Windows. The pinned versions in `requirements.txt` are tested and stable. If you still get a DLL error, install the [Microsoft Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe) and restart.

---

## Quick start

### Single experiment

Edit the `GMM0` / `GMM1` block at the top of `experiments/run_experiment.py`, then:

```bash
python experiments/run_experiment.py
```

This trains MCD, CMCD and LED, runs evaluation, prints a metrics table and saves plots to `results/`.

### Multi-experiment pipeline

Edit the `EXPERIMENTS` list at the top of `experiments/run_overnight.py` and run:

```bash
python experiments/run_overnight.py
```

The script automatically, for each experiment:
1. Plots the start and end distributions ($\pi_0$ and $\pi_1$)
2. Trains MCD, CMCD and LED (skips if `.pkl` already exist)
3. Evaluates all four algorithms
4. Saves `work_histograms.png`, `ratio_convergence.png`, `loss_curves.png`
5. Writes `metrics.txt` and a global `summary.txt`

### Defining your own distributions

Only the `EXPERIMENTS` block needs to be changed. Three constructors are available:

```python
from src.gmm_config import GMMParams
import numpy as np

# Single Gaussian
gmm0 = GMMParams.single_gaussian(mean=[-2.0], cov=[[1.0]])          # 1D
gmm0 = GMMParams.single_gaussian(mean=[0., 0.], cov=np.eye(2))      # 2D

# Multiple components, same isotropic covariance
gmm0 = GMMParams.isotropic(means=[[-3, 0], [3, 0]], var=1.0)

# Full control
gmm0 = GMMParams(
    means=np.array([[-4., 0.], [0., 4.]]),           # (K, d)
    covs=np.stack([np.eye(2), 0.5 * np.eye(2)]),     # (K, d, d)
    weights=np.array([0.6, 0.4]),                     # must sum to 1
)
```

---

## Experiments

Five benchmark experiments of increasing complexity:

| # | Problem | $d$ | $K_0$ | $K_1$ |
|---|---|---|---|---|
| 1 | Gaussian → Gaussian | 1 | 1 | 1 |
| 2 | GMM → GMM symmetric | 1 | 2 | 2 |
| 3 | Gaussian → Gaussian | 2 | 1 | 1 |
| 4 | GMM → GMM rotating modes | 2 | 2 | 2 |
| 5 | GMM → GMM asymmetric | 2 | 3 | 3 |

---

## Implementation notes

### Memory management
`jax.vmap` over 10,000 trajectories × 1,000 steps would allocate ~40 GB. All `estimate()` functions use a mini-batch loop (`eval_batch_size=256`) via `gmm_eval_utils.batched_estimate`, keeping memory bounded.

### Unnormalised densities
A subtle but critical point: the work formula (eq. 5.18) requires **unnormalised** log densities $\log\gamma_k(q)$, not normalised $\log\pi_k(q)$. Using normalised densities introduces an offset $\log(Z_1/Z_0)$ that makes the estimator converge to 1 instead of $Z_1/Z_0$. See `gmm_jax_physics.py` for details.

### Separate train/eval time steps
The network is trained with $K=64$ steps ($dt_\text{train} = T/64$) to avoid OOM and gradient explosion through long chains. Evaluation uses finer steps ($dt_\text{eval} = 10^{-3}$). The embedding index is mapped via $k_\text{train} = \lfloor k_\text{eval} \cdot dt_\text{eval} / dt_\text{train} \rfloor$.

### LED bias
The log-variance loss $\log(\text{Var}(W^\theta))$ achieves near-zero variance but does not anchor $\mathbb{E}[W^\theta]$ to $\Delta F$. This is a known limitation discussed in Section 5.5 of the thesis. A full fix requires the Poisson-problem algorithm (also described there).

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `jax` / `jaxlib` | 0.4.30 | JIT compilation, autodiff, vmap |
| `optax` | 0.2.3 | Adam optimizer, learning rate schedules |
| `numpy` | ≥1.22 | Array operations |
| `scipy` | ≥1.9 | GMM density evaluation, logsumexp |
| `matplotlib` | ≥3.6 | Plots |
| `tqdm` | ≥4.64 | Progress bars |

---

## Reference

This project is a reimplementation and extension of:

> Régis Santet. *Foundations and Optimization of Langevin Samplers.*  
> PhD thesis, École des Ponts ParisTech, 2024.  
> [tel-05162163](https://pastel.hal.science/tel-05162163v1)

Key algorithms:
- **MCD**: Doucet et al. (2022), *Score-based diffusion meets annealed importance sampling*
- **CMCD**: Vargas et al. (2023), *Denoising Diffusion Samplers*
- **LED**: Vaikuntanathan & Jarzynski (2009), *Escorted Free Energy Simulations*
- **Hutchinson divergence**: Hutchinson (1989), *A stochastic estimator of the trace*

---

## License

MIT License — see `LICENSE` for details.
