"""
gmm_metrics.py
──────────────
NaN-safe metrics and comparison table for the four algorithms.
"""

from __future__ import annotations
import numpy as np
import numpy.typing as npt
from typing import Dict


def compute_metrics(works: npt.NDArray) -> dict:
    w = np.asarray(works, dtype=float)
    w_finite = w[np.isfinite(w)]
    n_total  = len(w)
    n_finite = len(w_finite)
    n_nan    = n_total - n_finite

    if n_finite < 2:
        return {
            "ratio_estimate": float("nan"), "ratio_variance": float("nan"),
            "ratio_ci_half":  float("nan"), "dF_estimate":    float("nan"),
            "dF_variance":    float("nan"), "work_mean":      float("nan"),
            "work_variance":  float("nan"), "n_samples": n_total, "n_nan": n_nan,
        }

    e_mW      = np.exp(-w_finite)
    ratio     = float(e_mW.mean())
    ratio_var = float(e_mW.var(ddof=1))
    ci_half   = 1.96 * float(e_mW.std(ddof=1)) / np.sqrt(n_finite)
    dF        = float(-np.log(ratio + 1e-300))
    dF_var    = float(ratio_var / (ratio + 1e-300) ** 2)
    work_var  = float(w_finite.var(ddof=1))
    work_mean = float(w_finite.mean())

    return {
        "ratio_estimate": ratio, "ratio_variance": ratio_var,
        "ratio_ci_half":  ci_half, "dF_estimate": dF,
        "dF_variance":    dF_var, "work_mean":   work_mean,
        "work_variance":  work_var, "n_samples":  n_total, "n_nan": n_nan,
    }


def print_comparison_table(
    results: Dict[str, dict],
    reference_ratio: float | None = None,
) -> dict:
    cols  = ["Algorithm", "Ratio est.", "95% CI", "ΔF est.",
             "Var(ratio)", "Var(ΔF)", "Var(W)"]
    col_w = [22, 12, 12, 10, 14, 14, 12]
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
        m = compute_metrics(res["works"])
        all_metrics[name] = m

        def fmt(v, fmt_str=".6f"):
            return f"{v:{fmt_str}}" if np.isfinite(v) else "nan"

        row_vals = [
            name,
            fmt(m["ratio_estimate"]),
            f"±{fmt(m['ratio_ci_half'], '.4f')}",
            fmt(m["dF_estimate"], ".4f"),
            fmt(m["ratio_variance"], ".3e"),
            fmt(m["dF_variance"],   ".3e"),
            fmt(m["work_variance"], ".3e"),
        ]
        if m["n_nan"] > 0:
            row_vals[0] += f" ({m['n_nan']} NaN)"

        row = sep.join(v.ljust(w) for v, w in zip(row_vals, col_w))
        print(row)

    print(line)
    print()
    return all_metrics


def variance_reduction_table(all_metrics: dict, baseline: str = "AIS") -> None:
    if baseline not in all_metrics:
        print(f"Baseline '{baseline}' not in results.")
        return

    base_rv = all_metrics[baseline]["ratio_variance"]
    base_wv = all_metrics[baseline]["work_variance"]

    print(f"Variance reduction relative to {baseline}:")
    print(f"{'Algorithm':<22}  {'VR(ratio)':<12}  {'VR(work)':<12}")
    print("-" * 50)
    for name, m in all_metrics.items():
        rv = m["ratio_variance"]
        wv = m["work_variance"]
        vr_r = base_rv / rv  if np.isfinite(rv) and rv > 0 else float("nan")
        vr_w = base_wv / wv if np.isfinite(wv) and wv > 0 else float("nan")
        r_str = f"{vr_r:.2f}" if np.isfinite(vr_r) else "nan"
        w_str = f"{vr_w:.2f}" if np.isfinite(vr_w) else "nan"
        print(f"{name:<22}  {r_str:<12}  {w_str:<12}")
    print()
