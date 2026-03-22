"""
gmm_plots.py
────────────
Matplotlib visualisation helpers for the GMM comparison pipeline.
All plots are NaN-safe: algorithms with unstable works are flagged,
not crashed over.
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import Dict


PALETTE = [
    "#1f77b4", "#d62728", "#2ca02c", "#ff7f0e",
    "#9467bd", "#8c564b", "#e377c2", "#17becf",
]


def plot_work_histograms(
    results: Dict[str, dict],
    reference_dF: float | None = None,
    save_path: Path | None = None,
    title: str = "Work histograms",
):
    names = list(results.keys())
    n     = len(names)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5 * ncols, 4 * nrows),
                             squeeze=False)
    fig.suptitle(title, fontsize=14, fontweight="bold")

    for idx, name in enumerate(names):
        ax    = axes[idx // ncols][idx % ncols]
        w     = np.asarray(results[name]["works"], dtype=float)
        color = PALETTE[idx % len(PALETTE)]

        # Drop NaN / Inf
        w_finite = w[np.isfinite(w)]
        n_nan    = len(w) - len(w_finite)

        if len(w_finite) < 2:
            ax.text(0.5, 0.5, "No finite works\n(training unstable)",
                    ha="center", va="center", transform=ax.transAxes,
                    color="red", fontsize=12)
            ax.set_title(f"{name}  ⚠ NaN", fontsize=11)
            ax.set_xlabel("W"); ax.set_ylabel("Density")
            continue

        label_w = f"W  ({n_nan} NaN dropped)" if n_nan > 0 else "W"
        ax.hist(w_finite, bins=50, color=color, alpha=0.75,
                density=True, label=label_w)

        ratio_est = np.mean(np.exp(-w_finite))
        dF_est    = float(-np.log(ratio_est + 1e-300))
        ax.axvline(dF_est, color="red", ls="--", lw=1.5,
                   label=f"ΔF est. = {dF_est:.3f}")

        if reference_dF is not None:
            ax.axvline(reference_dF, color="black", ls="-", lw=1.5,
                       label=f"ΔF ref. = {reference_dF:.3f}")

        ax.set_title(name, fontsize=11)
        ax.set_xlabel("W"); ax.set_ylabel("Density")
        ax.legend(fontsize=8)

    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  → {save_path}")
    plt.show()


def plot_loss_curves(
    loss_histories: Dict[str, list],
    reference_dF: float | None = None,
    save_path: Path | None = None,
):
    fig, ax = plt.subplots(figsize=(8, 4))

    for idx, (name, losses) in enumerate(loss_histories.items()):
        losses_arr = np.asarray(losses, dtype=float)
        ax.plot(losses_arr, label=name,
                color=PALETTE[idx % len(PALETTE)], lw=1.5)

    if reference_dF is not None:
        ax.axhline(reference_dF, color="black", ls="--", lw=1.5,
                   label=f"ΔF reference = {reference_dF:.3f}")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss  (−ELBO ≈ E[W])")
    ax.set_title("Training losses")
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  → {save_path}")
    plt.show()


def plot_ratio_convergence(
    results: Dict[str, dict],
    reference_ratio: float | None = None,
    save_path: Path | None = None,
):
    fig, ax = plt.subplots(figsize=(9, 5))

    for idx, (name, res) in enumerate(results.items()):
        w = np.asarray(res["works"], dtype=float)
        w_finite = w[np.isfinite(w)]
        if len(w_finite) < 2:
            continue
        cum_m = np.cumsum(np.exp(-w_finite)) / np.arange(1, len(w_finite) + 1)
        ax.plot(cum_m, label=name,
                color=PALETTE[idx % len(PALETTE)], lw=1.5)

    if reference_ratio is not None:
        ax.axhline(reference_ratio, color="black", ls="--", lw=1.5,
                   label=f"Reference = {reference_ratio:.4f}")

    ax.set_xlabel("Number of samples")
    ax.set_ylabel("Running mean  E[exp(−W)]")
    ax.set_title("Convergence of ratio estimator")
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  → {save_path}")
    plt.show()


def plot_2d_trajectories(
    results: Dict[str, dict],
    n_display: int = 30,
    save_path: Path | None = None,
):
    fig = plt.figure(figsize=(6 * len(results), 5))
    gs  = gridspec.GridSpec(1, len(results))

    for idx, (name, res) in enumerate(results.items()):
        trajs = res.get("trajs")
        if trajs is None or trajs.shape[-1] != 2:
            continue
        ax = fig.add_subplot(gs[idx])
        n_traj = min(n_display, trajs.shape[0])
        for i in range(n_traj):
            ax.plot(trajs[i, :, 0], trajs[i, :, 1],
                    alpha=0.4, lw=0.8, color=PALETTE[idx % len(PALETTE)])
            ax.scatter(*trajs[i, 0],  s=20, color="green", zorder=5)
            ax.scatter(*trajs[i, -1], s=20, color="red",   zorder=5)
        ax.set_title(name)
        ax.set_xlabel("q₁"); ax.set_ylabel("q₂")

    plt.suptitle("Sample trajectories (green=start, red=end)", fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  → {save_path}")
    plt.show()
