"""
gmm_plots.py
============
Visualisation functions for the simulation results.

THREE TYPES OF PLOTS
--------------------

1. WORK HISTOGRAMS (plot_work_histograms)
   Shows the distribution of virtual works W_i for each algorithm.

   WHAT TO LOOK FOR:
   - The distribution should be centred near Delta_F (the true free energy).
   - Narrow distribution = low variance = good estimator.
   - Wide distribution = high variance = noisy estimator (AIS).
   - The red dashed line shows our estimate of Delta_F.
   - The black line shows the true Delta_F (if known).

   A PERFECT algorithm would have all W_i = Delta_F exactly (zero variance):
   the histogram would be a single spike at Delta_F.

2. LOSS CURVES (plot_loss_curves)
   Shows how the training loss decreases over epochs for MCD, CMCD and LED.

   WHAT TO LOOK FOR:
   - The loss should decrease smoothly towards the reference line (Delta_F).
   - If the loss oscillates wildly, the learning rate is too high.
   - If the loss plateaus far above Delta_F, the network has not converged.
   - The black dashed line is Delta_F — the theoretical minimum for the loss.

   WHY DOES THE LOSS CONVERGE TO Delta_F?
   At optimum, the ELBO loss = E[-ELBO] = E[W] which should equal Delta_F.
   (For LED, the loss is Var(W) which should converge to 0, but we plot E[W]
   for comparability.)

3. RATIO CONVERGENCE (plot_ratio_convergence)
   Shows how the running mean E[exp(-W_i)] evolves as we collect more samples.

   WHAT TO LOOK FOR:
   - All curves should converge to the true Z1/Z0 (black dashed line).
   - Smoother convergence = lower variance = better algorithm.
   - AIS typically shows large jumps (rare events dominate the average).
   - MCD, CMCD, LED should converge more smoothly.

NaN SAFETY
----------
All plotting functions silently handle NaN/Inf works:
- NaN values are dropped before plotting
- A warning is shown if too many values are NaN
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import Dict


# Colour palette for distinguishing algorithms in plots
# These colours are chosen to be distinct and colourblind-friendly
PALETTE = [
    "#1f77b4",  # blue    -> AIS
    "#d62728",  # red     -> MCD
    "#2ca02c",  # green   -> CMCD
    "#ff7f0e",  # orange  -> LED
    "#9467bd",  # purple  -> (extra algorithms if any)
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#17becf",  # cyan
]


def plot_work_histograms(
    results: Dict[str, dict],
    reference_dF: float | None = None,
    save_path: Path | None = None,
    title: str = "Work histograms",
):
    """
    Plot the distribution of virtual works W_i for each algorithm.

    Each subplot shows one algorithm. The x-axis is the work W,
    the y-axis is the probability density.

    A GOOD algorithm produces a narrow, well-centred histogram:
    - Centred at Delta_F (works have correct mean)
    - Narrow (low variance)

    AIS typically gives a wide right-skewed distribution.
    MCD and CMCD give narrower distributions.
    LED can give an extremely narrow distribution but sometimes biased.

    Parameters
    ----------
    results : dict
        Keys are algorithm names, values have a "works" key with the work arrays.
        Direct output of run() / estimate() from each algorithm.

    reference_dF : float or None
        True value of Delta_F if known analytically.
        Shown as a black vertical line for comparison.

    save_path : Path or None
        If given, save the figure to this file path (PNG format).
        If None, only display interactively.

    title : str
        Overall figure title.
    """
    names = list(results.keys())
    n     = len(names)
    ncols = min(3, n)   # at most 3 columns per row
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5 * ncols, 4 * nrows),
                             squeeze=False)
    fig.suptitle(title, fontsize=14, fontweight="bold")

    for idx, name in enumerate(names):
        ax    = axes[idx // ncols][idx % ncols]
        w     = np.asarray(results[name]["works"], dtype=float)
        color = PALETTE[idx % len(PALETTE)]

        # Remove NaN and Inf values (can occur when training is unstable)
        w_finite = w[np.isfinite(w)]
        n_nan    = len(w) - len(w_finite)

        # If almost everything is NaN, show a warning instead of a plot
        if len(w_finite) < 2:
            ax.text(0.5, 0.5, "No finite works\n(training unstable)",
                    ha="center", va="center", transform=ax.transAxes,
                    color="red", fontsize=12)
            ax.set_title(f"{name}  WARNING: NaN", fontsize=11)
            ax.set_xlabel("W")
            ax.set_ylabel("Density")
            continue

        # Label: mention if some NaN values were dropped
        label_w = f"W  ({n_nan} NaN dropped)" if n_nan > 0 else "W"

        # Plot the histogram as a probability density (area = 1)
        ax.hist(w_finite, bins=50, color=color, alpha=0.75,
                density=True, label=label_w)

        # Compute and show the Delta_F estimate from this algorithm
        # Jarzynski: Z1/Z0 = mean(exp(-W)), so Delta_F = -log(mean(exp(-W)))
        ratio_est = np.mean(np.exp(-w_finite))
        dF_est    = float(-np.log(ratio_est + 1e-300))
        ax.axvline(dF_est, color="red", ls="--", lw=1.5,
                   label=f"dF est. = {dF_est:.3f}")

        # Show the true Delta_F for comparison (if known)
        if reference_dF is not None:
            ax.axvline(reference_dF, color="black", ls="-", lw=1.5,
                       label=f"dF ref. = {reference_dF:.3f}")

        ax.set_title(name, fontsize=11)
        ax.set_xlabel("W")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)

    # Hide any empty subplots (if n is not a multiple of ncols)
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  -> {save_path}")
    plt.show()


def plot_loss_curves(
    loss_histories: Dict[str, list],
    reference_dF: float | None = None,
    save_path: Path | None = None,
):
    """
    Plot the training loss as a function of epoch for MCD, CMCD and LED.

    The y-axis shows the loss value, which approximates E[W] (the expected
    virtual work). At the end of training, this should be close to Delta_F.

    The black dashed line shows the theoretical target Delta_F (if known).

    INTERPRETING THE CURVES:
    - Decreasing curve: training is working, the network is improving.
    - Flat curve: the network has converged (or is stuck at a bad local minimum).
    - Oscillating curve: learning rate may be too high.
    - Loss below Delta_F: possible overfitting or degeneracy (especially LED).

    Parameters
    ----------
    loss_histories : dict
        Keys are algorithm names ("MCD", "CMCD", "LED").
        Values are lists of loss values, one per epoch.
        These are returned by the train() function of each algorithm.

    reference_dF : float or None
        The true Delta_F shown as a horizontal dashed line.

    save_path : Path or None
        If given, save the figure to this file.
    """
    fig, ax = plt.subplots(figsize=(8, 4))

    for idx, (name, losses) in enumerate(loss_histories.items()):
        losses_arr = np.asarray(losses, dtype=float)
        ax.plot(losses_arr, label=name,
                color=PALETTE[idx % len(PALETTE)], lw=1.5)

    # Show the target value Delta_F that a perfect algorithm would achieve
    if reference_dF is not None:
        ax.axhline(reference_dF, color="black", ls="--", lw=1.5,
                   label=f"Delta_F reference = {reference_dF:.3f}")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss  (-ELBO, approx. E[W])")
    ax.set_title("Training losses")
    ax.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  -> {save_path}")
    plt.show()


def plot_ratio_convergence(
    results: Dict[str, dict],
    reference_ratio: float | None = None,
    save_path: Path | None = None,
):
    """
    Plot the running mean of exp(-W_i) as a function of the number of samples.

    WHAT THIS SHOWS:
    After collecting i samples, the running estimate of Z1/Z0 is:
        running_mean_i = (1/i) * sum_{j=1}^{i} exp(-W_j)

    This plot shows how this running mean evolves as i goes from 1 to N.

    A GOOD algorithm:
    - Converges quickly to the true Z1/Z0 (black dashed line)
    - Has a smooth curve (low fluctuations = low variance)
    - Requires few samples to reach a stable value

    AIS: noisy, slow convergence, large jumps when rare trajectories appear
    MCD/CMCD: smoother, faster convergence
    LED: very smooth but may converge to a wrong value (bias)

    Parameters
    ----------
    results : dict
        Same format as in plot_work_histograms.

    reference_ratio : float or None
        True value of Z1/Z0 shown as a black dashed line.

    save_path : Path or None
        If given, save the figure to this file.
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    for idx, (name, res) in enumerate(results.items()):
        w = np.asarray(res["works"], dtype=float)
        w_finite = w[np.isfinite(w)]

        if len(w_finite) < 2:
            continue  # skip algorithms with no finite works

        # Compute running mean: running_mean[i] = mean(exp(-W_1), ..., exp(-W_i))
        # np.cumsum computes cumulative sums: [a1, a1+a2, a1+a2+a3, ...]
        # dividing by [1, 2, 3, ...] gives the running mean
        cum_m = np.cumsum(np.exp(-w_finite)) / np.arange(1, len(w_finite) + 1)

        ax.plot(cum_m, label=name,
                color=PALETTE[idx % len(PALETTE)], lw=1.5)

    if reference_ratio is not None:
        ax.axhline(reference_ratio, color="black", ls="--", lw=1.5,
                   label=f"Reference = {reference_ratio:.4f}")

    ax.set_xlabel("Number of samples")
    ax.set_ylabel("Running mean  E[exp(-W)]")
    ax.set_title("Convergence of ratio estimator")
    ax.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  -> {save_path}")
    plt.show()


def plot_2d_trajectories(
    results: Dict[str, dict],
    n_display: int = 30,
    save_path: Path | None = None,
):
    """
    Plot sample Langevin trajectories in 2D configuration space.

    Only works for d=2 problems. Shows the paths of particles as they
    evolve from pi_0 (green dots) to pi_1 (red dots).

    WHAT TO LOOK FOR:
    - Trajectories should generally move from the region of high pi_0 density
      towards the region of high pi_1 density.
    - Well-trained algorithms produce more direct, less noisy paths.

    Parameters
    ----------
    results    : dict — algorithm results (must have "trajs" key with shape (N, K, 2))
    n_display  : int — how many trajectories to show (default 30)
    save_path  : Path or None
    """
    fig = plt.figure(figsize=(6 * len(results), 5))
    gs  = gridspec.GridSpec(1, len(results))

    for idx, (name, res) in enumerate(results.items()):
        trajs = res.get("trajs")

        # Skip if trajectories were not stored or not 2D
        if trajs is None or trajs.shape[-1] != 2:
            continue

        ax     = fig.add_subplot(gs[idx])
        n_traj = min(n_display, trajs.shape[0])   # don't show more than n_display

        for i in range(n_traj):
            # Plot the trajectory as a line
            ax.plot(trajs[i, :, 0], trajs[i, :, 1],
                    alpha=0.4, lw=0.8, color=PALETTE[idx % len(PALETTE)])
            # Mark start (green) and end (red)
            ax.scatter(*trajs[i, 0],  s=20, color="green", zorder=5)
            ax.scatter(*trajs[i, -1], s=20, color="red",   zorder=5)

        ax.set_title(name)
        ax.set_xlabel("q1")
        ax.set_ylabel("q2")

    plt.suptitle("Sample trajectories (green=start, red=end)", fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  -> {save_path}")
    plt.show()
