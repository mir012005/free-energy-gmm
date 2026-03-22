"""
gmm_metrics.py
==============
Statistical metrics and comparison tables for the four algorithms.

WHAT WE MEASURE
---------------
Each algorithm returns N virtual works  W_1, W_2, ..., W_N.
From these works, we want to estimate:

    1. Z1/Z0  (the partition function ratio = the "answer" we want)
       Estimator: mean(exp(-W_i))   by Jarzynski's equality

    2. Delta_F  (the free energy difference)
       Estimator: -log(Z1/Z0)  =  -log(mean(exp(-W_i)))

    3. The QUALITY of the estimator (how reliable is our answer?)
       Measured by: variance of the estimator

VARIANCE — THE KEY QUALITY METRIC
-----------------------------------
Variance tells us how much the estimator fluctuates from run to run.
Low variance means the algorithm gives consistent results with few samples.
High variance means we need many more samples to get a reliable answer.

We report three variance-related quantities:

    Var(ratio) = variance of exp(-W_i) across trajectories
                 This is the variance of our basic estimator.
                 Ideal: close to 0.

    Var(Delta_F) = variance of the Delta_F estimator
                   Approximated via error propagation:
                   Var(Delta_F) ≈ Var(ratio) / (mean(exp(-W_i)))^2

    Var(W)      = variance of the works W_i themselves
                  This measures how spread out the work distribution is.
                  If all trajectories give the same work (W_i = Delta_F for all i),
                  then Var(W) = 0 and the algorithm is "zero-variance".

    VR (variance reduction) = Var_AIS / Var_algorithm
                              How many times better than AIS.
                              VR = 100 means 100x less variance than AIS,
                              i.e. you need 100x fewer samples to get the
                              same statistical accuracy.

NaN SAFETY
----------
Some algorithms can produce NaN (Not a Number) or Inf (Infinity) works
when training is unstable (exploding gradients, etc.).
All functions here filter these out before computing statistics and
report the number of dropped values so the user knows.

95% CONFIDENCE INTERVAL
------------------------
The confidence interval gives a range [estimate - half_width, estimate + half_width]
such that there is a 95% probability the true value lies within.
It is computed as: 1.96 * std(exp(-W_i)) / sqrt(N)
(standard normal approximation, valid for large N by the Central Limit Theorem)
"""

from __future__ import annotations
import numpy as np
import numpy.typing as npt
from typing import Dict


def compute_metrics(works: npt.NDArray) -> dict:
    """
    Compute all statistical metrics from an array of virtual works.

    Parameters
    ----------
    works : array of shape (n_samples,)
            The virtual works W_1, ..., W_N from one algorithm.
            May contain NaN or Inf values (these are filtered out).

    Returns
    -------
    dict with the following keys:

        ratio_estimate : float
            The estimated Z1/Z0 = mean(exp(-W_i)).
            Should be close to the true Z1/Z0.

        ratio_variance : float
            Var(exp(-W_i)) — variance of the ratio estimator.
            Lower is better.

        ratio_ci_half : float
            Half-width of the 95% confidence interval on ratio_estimate.
            The true Z1/Z0 lies in [ratio_estimate ± ratio_ci_half]
            with 95% probability.

        dF_estimate : float
            Estimated free energy difference Delta_F = -log(ratio_estimate).

        dF_variance : float
            Variance of the Delta_F estimator (via error propagation).

        work_mean : float
            Mean of the works: mean(W_i).
            For a well-trained algorithm this should be close to Delta_F.

        work_variance : float
            Variance of the works: Var(W_i).
            Zero would mean all trajectories give identical work = Delta_F.

        n_samples : int
            Total number of trajectories (including NaN ones).

        n_nan : int
            Number of trajectories with NaN or Inf work.
            Should be 0 for a stable algorithm.
    """
    w        = np.asarray(works, dtype=float)
    w_finite = w[np.isfinite(w)]    # keep only finite (non-NaN, non-Inf) values
    n_total  = len(w)
    n_finite = len(w_finite)
    n_nan    = n_total - n_finite   # how many were dropped

    # If too few finite works, return NaN for everything
    if n_finite < 2:
        return {
            "ratio_estimate": float("nan"),
            "ratio_variance": float("nan"),
            "ratio_ci_half":  float("nan"),
            "dF_estimate":    float("nan"),
            "dF_variance":    float("nan"),
            "work_mean":      float("nan"),
            "work_variance":  float("nan"),
            "n_samples":      n_total,
            "n_nan":          n_nan,
        }

    # --- Compute the ratio estimator ---
    e_mW      = np.exp(-w_finite)                           # exp(-W_i), shape (n_finite,)
    ratio     = float(e_mW.mean())                          # mean(exp(-W_i)) = Z1/Z0
    ratio_var = float(e_mW.var(ddof=1))                     # sample variance of exp(-W_i)

    # 95% confidence interval half-width using the normal approximation
    # std / sqrt(N) is the standard error of the mean
    ci_half = 1.96 * float(e_mW.std(ddof=1)) / np.sqrt(n_finite)

    # --- Free energy difference ---
    dF      = float(-np.log(ratio + 1e-300))                # Delta_F = -log(Z1/Z0)
    # 1e-300 prevents log(0) if ratio is accidentally exactly 0

    # Delta_F variance via the delta method (error propagation):
    # Var(f(x)) ≈ (f'(x))^2 * Var(x)
    # f(x) = -log(x), f'(x) = -1/x
    # So Var(Delta_F) ≈ (1/ratio)^2 * Var(ratio)
    dF_var  = float(ratio_var / (ratio + 1e-300) ** 2)

    # --- Work statistics ---
    work_var  = float(w_finite.var(ddof=1))
    work_mean = float(w_finite.mean())

    return {
        "ratio_estimate": ratio,
        "ratio_variance": ratio_var,
        "ratio_ci_half":  ci_half,
        "dF_estimate":    dF,
        "dF_variance":    dF_var,
        "work_mean":      work_mean,
        "work_variance":  work_var,
        "n_samples":      n_total,
        "n_nan":          n_nan,
    }


def print_comparison_table(
    results: Dict[str, dict],
    reference_ratio: float | None = None,
) -> dict:
    """
    Print a formatted comparison table of all algorithms and return the metrics.

    For each algorithm, shows:
        - Ratio estimate (should be close to the true Z1/Z0)
        - 95% confidence interval
        - Delta_F estimate
        - Var(ratio), Var(Delta_F), Var(W)

    Parameters
    ----------
    results : dict
        Keys are algorithm names ("AIS", "MCD", etc.).
        Values are dicts with key "works" containing the work arrays.
        This is the direct output of run() or estimate() from each algorithm.

    reference_ratio : float or None
        The known true value of Z1/Z0 (if available analytically).
        Printed for comparison. Use None if the true value is unknown.

    Returns
    -------
    dict
        Keys are algorithm names.
        Values are metric dicts from compute_metrics().
        Useful for further analysis (variance reduction, plots, etc.).
    """
    # Column names and widths for the formatted table
    cols  = ["Algorithm", "Ratio est.", "95% CI", "Delta_F est.",
             "Var(ratio)", "Var(Delta_F)", "Var(W)"]
    col_w = [22, 12, 12, 12, 14, 14, 12]
    sep   = "  "

    header = sep.join(c.ljust(w) for c, w in zip(cols, col_w))
    line   = "-" * len(header)

    print()
    print(line)
    print(header)
    print(line)
    if reference_ratio is not None:
        print(f"  (reference Z1/Z0 = {reference_ratio:.6f})")
    print(line)

    all_metrics = {}

    for name, res in results.items():
        # Compute metrics for this algorithm
        m = compute_metrics(res["works"])
        all_metrics[name] = m

        def fmt(v, fmt_str=".6f"):
            """Format a float, or show 'nan' if not finite."""
            return f"{v:{fmt_str}}" if np.isfinite(v) else "nan"

        # Build the row values
        row_vals = [
            name + (f" ({m['n_nan']} NaN)" if m["n_nan"] > 0 else ""),
            fmt(m["ratio_estimate"]),
            f"+-{fmt(m['ratio_ci_half'], '.4f')}",
            fmt(m["dF_estimate"], ".4f"),
            fmt(m["ratio_variance"], ".3e"),
            fmt(m["dF_variance"],   ".3e"),
            fmt(m["work_variance"], ".3e"),
        ]

        row = sep.join(v.ljust(w) for v, w in zip(row_vals, col_w))
        print(row)

    print(line)
    print()
    return all_metrics


def variance_reduction_table(all_metrics: dict, baseline: str = "AIS") -> None:
    """
    Print the variance reduction (VR) of each algorithm relative to AIS.

    Variance Reduction (VR) = Var_baseline / Var_algorithm

    Interpretation:
        VR = 1    : same variance as AIS (no improvement)
        VR = 100  : 100x less variance than AIS
                    -> need 100x fewer samples for the same accuracy
        VR = 1000 : 1000x less variance -> 1000x fewer samples needed

    High VR is good: it means the algorithm is much more efficient than AIS.

    Parameters
    ----------
    all_metrics : dict
        Output of print_comparison_table().

    baseline : str, default "AIS"
        The algorithm to use as the reference for VR computation.
    """
    if baseline not in all_metrics:
        print(f"Baseline '{baseline}' not found in results.")
        return

    base_rv = all_metrics[baseline]["ratio_variance"]
    base_wv = all_metrics[baseline]["work_variance"]

    print(f"Variance reduction relative to {baseline}:")
    print(f"{'Algorithm':<22}  {'VR(ratio)':<12}  {'VR(work)':<12}")
    print("-" * 50)

    for name, m in all_metrics.items():
        rv = m["ratio_variance"]
        wv = m["work_variance"]

        # VR = baseline_variance / algorithm_variance
        # If the algorithm variance is 0 (perfect), VR = infinity
        vr_r = base_rv / rv  if (np.isfinite(rv) and rv > 0) else float("nan")
        vr_w = base_wv / wv  if (np.isfinite(wv) and wv > 0) else float("nan")

        r_str = f"{vr_r:.2f}" if np.isfinite(vr_r) else "nan"
        w_str = f"{vr_w:.2f}" if np.isfinite(vr_w) else "nan"

        print(f"{name:<22}  {r_str:<12}  {w_str:<12}")

    print()
