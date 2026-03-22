"""
run_overnight.py
================
Pipeline complete pour comparer AIS / MCD / CMCD / LED sur des GMMs.

UTILISATION
-----------
    1. Modifier uniquement le bloc EXPERIENCES ci-dessous
    2. Lancer : python run_overnight.py

Le script fait automatiquement pour chaque experience :
    - Visualisation des distributions pi_0 et pi_1
    - Entrainement MCD / CMCD / LED  (saute si .pkl deja present)
    - Evaluation AIS / MCD / CMCD / LED
    - Metriques et plots (histogrammes, convergence, loss curves)
    - Sauvegarde dans results/<nom_experience>/

A la fin : results/summary.txt  (rapport comparatif toutes experiences)

STRUCTURE DES RESULTATS
-----------------------
results/
+-- exp1_xxx/
|   +-- distributions.png     <- pi_0 et pi_1
|   +-- work_histograms.png
|   +-- ratio_convergence.png
|   +-- loss_curves.png
|   +-- metrics.txt
|   +-- mcd_params.pkl  mcd_meta.pkl
|   +-- cmcd_params.pkl cmcd_meta.pkl
|   +-- led_params.pkl  led_meta.pkl
+-- exp2_xxx/
+-- ...
+-- distributions_overview.png  <- toutes les experiences sur une image
+-- summary.txt                 <- tableau comparatif final
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import pickle
import traceback
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from datetime import datetime
from scipy.stats import multivariate_normal

from gmm_config import GMMParams, PipelineConfig
import gmm_ais, gmm_mcd, gmm_cmcd, gmm_led
from gmm_metrics import compute_metrics, print_comparison_table, variance_reduction_table
from gmm_plots import plot_work_histograms, plot_loss_curves, plot_ratio_convergence


# =============================================================================
#
#   EXPERIENCES — MODIFIER UNIQUEMENT CE BLOC
#
#   Chaque experience est un dict avec :
#     "name"  : identifiant du dossier (pas d'espaces ni de caracteres speciaux)
#     "label" : titre lisible pour les plots
#     "gmm0"  : distribution de depart   (GMMParams)
#     "gmm1"  : distribution d'arrivee   (GMMParams)
#     "T"     : temps total de simulation (plus les distributions sont eloignees,
#                plus T doit etre grand pour laisser la dynamique converger)
#     "ref"   : Z1/Z0 analytique si connu, None sinon (calcule par MC)
#
#   Constructeurs disponibles pour GMMParams :
#     GMMParams.single_gaussian(mean, cov)
#         ex : GMMParams.single_gaussian([-2.0], [[1.0]])          <- 1D
#         ex : GMMParams.single_gaussian([0.0, 0.0], np.eye(2))    <- 2D
#
#     GMMParams.isotropic(means, var, weights=None)
#         ex : GMMParams.isotropic([[-3,0],[3,0]], var=1.0)         <- 2 composantes, meme variance
#
#     GMMParams(means, covs, weights)                               <- controle total
#         means   : (K, d)    une ligne par composante
#         covs    : (K, d, d) une matrice de covariance par composante
#         weights : (K,)      poids (doivent sommer a 1)
#
# =============================================================================

EXPERIMENTS = [

    # -- Experience 1 : 1D Gaussienne -> Gaussienne (these S5.4.1 baseline) --
    {
        "name":  "exp1_1d_gaussians",
        "label": "Exp 1 — 1D  N(-2,1) -> N(2,0.25)",
        "gmm0":  GMMParams.single_gaussian([-2.0], [[1.0]]),
        "gmm1":  GMMParams.single_gaussian([ 2.0], [[0.25]]),
        "T":     1.0,
        "ref":   0.5,    # Z1/Z0 = sqrt(0.25/1.0) = 0.5  analytique
    },

    # -- Experience 2 : 1D GMM(2) -> GMM(2) ----------------------------------
    {
        "name":  "exp2_1d_gmm2",
        "label": "Exp 2 — 1D  GMM(2) -> GMM(2)  symetrique",
        "gmm0":  GMMParams(
            means=np.array([[-4.0], [4.0]]),
            covs=np.array([[[1.0]], [[1.0]]]),
            weights=np.array([0.5, 0.5]),
        ),
        "gmm1":  GMMParams(
            means=np.array([[-2.0], [2.0]]),
            covs=np.array([[[0.25]], [[0.25]]]),
            weights=np.array([0.5, 0.5]),
        ),
        "T":     2.0,
        "ref":   None,
    },

    # -- Experience 3 : 2D Gaussienne -> Gaussienne ---------------------------
    {
        "name":  "exp3_2d_gaussians",
        "label": "Exp 3 — 2D  N([-1.5,-1.5], I) -> N([1.5,1.5], 0.5I)",
        "gmm0":  GMMParams.single_gaussian(np.array([-1.5, -1.5]), np.eye(2)),
        "gmm1":  GMMParams.single_gaussian(np.array([ 1.5,  1.5]), 0.5 * np.eye(2)),
        "T":     3.0,
        "ref":   None,
    },

    # -- Experience 4 : 2D GMM(2) -> GMM(2) modes qui tournent ---------------
    {
        "name":  "exp4_2d_gmm2",
        "label": "Exp 4 — 2D  GMM(2) -> GMM(2)  modes qui tournent",
        "gmm0":  GMMParams(
            means=np.array([[-4.0, 0.0], [0.0, 4.0]]),
            covs=np.stack([np.eye(2), np.eye(2)]),
            weights=np.array([0.5, 0.5]),
        ),
        "gmm1":  GMMParams(
            means=np.array([[4.0, 0.0], [0.0, -4.0]]),
            covs=np.stack([0.5 * np.eye(2), 0.5 * np.eye(2)]),
            weights=np.array([0.5, 0.5]),
        ),
        "T":     3.0,
        "ref":   None,
    },

    # -- Experience 5 : 2D GMM(3) -> GMM(3) asymetrique (difficile) ----------
    {
        "name":  "exp5_2d_gmm3_hard",
        "label": "Exp 5 — 2D  GMM(3) -> GMM(3)  asymetrique",
        "gmm0":  GMMParams(
            means=np.array([[-5.0, 0.0], [2.0, 4.0], [2.0, -4.0]]),
            covs=np.stack([np.eye(2), 0.5*np.eye(2), 0.5*np.eye(2)]),
            weights=np.array([0.5, 0.25, 0.25]),
        ),
        "gmm1":  GMMParams(
            means=np.array([[5.0, 0.0], [-2.0, -4.0], [-2.0, 4.0]]),
            covs=np.stack([0.25*np.eye(2), np.eye(2), np.eye(2)]),
            weights=np.array([0.6, 0.2, 0.2]),
        ),
        "T":     4.0,
        "ref":   None,
    },
]

# =============================================================================
#   FIN DU BLOC A MODIFIER
# =============================================================================


# =============================================================================
# HYPERPARAMETRES GLOBAUX  (modifier si necessaire)
# =============================================================================

DT_EVAL         = 1e-3
N_SAMPLES       = 10_000
EVAL_BATCH_SIZE = 256
SEED            = 42

K_TRAIN        = 64
N_EPOCHS_MAX   = 5000
EARLY_STOP_PAT = 1000
BATCH_SIZE     = 128
EMB_DIM        = 20

BASE_OUTPUT = Path("./results")
BASE_OUTPUT.mkdir(exist_ok=True)


# =============================================================================
# VISUALISATION DES DISTRIBUTIONS
# =============================================================================

def _gmm_density_1d(gmm: GMMParams, xs: np.ndarray) -> np.ndarray:
    density = np.zeros_like(xs)
    for k in range(gmm.n_components):
        mu  = gmm.means[k, 0]
        sig = float(np.sqrt(gmm.covs[k, 0, 0]))
        density += gmm.weights[k] * multivariate_normal.pdf(xs, mean=mu, cov=sig**2)
    return density


def _gmm_density_2d(gmm: GMMParams, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    pos = np.stack([X, Y], axis=-1)
    density = np.zeros(X.shape)
    for k in range(gmm.n_components):
        density += gmm.weights[k] * multivariate_normal.pdf(
            pos, mean=gmm.means[k], cov=gmm.covs[k])
    return density


def _plot_1d_ax(ax, gmm0, gmm1, label, ref):
    all_means = np.concatenate([gmm0.means[:, 0], gmm1.means[:, 0]])
    all_stds  = np.concatenate([
        np.sqrt(gmm0.covs[:, 0, 0]),
        np.sqrt(gmm1.covs[:, 0, 0]),
    ])
    margin = 4 * all_stds.max()
    xs = np.linspace(all_means.min() - margin, all_means.max() + margin, 500)

    d0 = _gmm_density_1d(gmm0, xs)
    d1 = _gmm_density_1d(gmm1, xs)

    ax.fill_between(xs, d0, alpha=0.30, color="#1f77b4")
    ax.plot(xs, d0, color="#1f77b4", lw=2,
            label=f"pi_0  ({gmm0.n_components} comp.)")
    ax.fill_between(xs, d1, alpha=0.30, color="#d62728")
    ax.plot(xs, d1, color="#d62728", lw=2,
            label=f"pi_1  ({gmm1.n_components} comp.)")

    for mu in gmm0.means[:, 0]:
        ax.axvline(mu, color="#1f77b4", ls="--", lw=0.8, alpha=0.5)
    for mu in gmm1.means[:, 0]:
        ax.axvline(mu, color="#d62728", ls="--", lw=0.8, alpha=0.5)

    ref_str = f"   Z1/Z0 = {ref:.3f}" if ref is not None else ""
    ax.set_title(label + ref_str, fontsize=9, fontweight="bold")
    ax.set_xlabel("q");  ax.set_ylabel("Densite")
    ax.legend(fontsize=8)


def _plot_2d_pair(ax0, ax1, gmm0, gmm1):
    all_means = np.concatenate([gmm0.means, gmm1.means], axis=0)
    all_stds  = np.concatenate([
        np.sqrt(np.array([gmm0.covs[k].diagonal() for k in range(gmm0.n_components)])),
        np.sqrt(np.array([gmm1.covs[k].diagonal() for k in range(gmm1.n_components)])),
    ])
    margin = 3.5 * all_stds.max()
    x0, x1 = all_means[:, 0].min() - margin, all_means[:, 0].max() + margin
    y0, y1 = all_means[:, 1].min() - margin, all_means[:, 1].max() + margin

    N  = 200
    X, Y = np.meshgrid(np.linspace(x0, x1, N), np.linspace(y0, y1, N))
    Z0 = _gmm_density_2d(gmm0, X, Y)
    Z1 = _gmm_density_2d(gmm1, X, Y)
    vmax = max(Z0.max(), Z1.max())

    for ax, Z, gmm, cmap, col, name in [
        (ax0, Z0, gmm0, "Blues", "#1f77b4", "pi_0  (depart)"),
        (ax1, Z1, gmm1, "Reds",  "#d62728", "pi_1  (arrivee)"),
    ]:
        im = ax.contourf(X, Y, Z, levels=20, cmap=cmap,
                         vmin=0, vmax=vmax, alpha=0.85)
        ax.contour(X, Y, Z, levels=6, colors=col, linewidths=0.8, alpha=0.6)
        for k in range(gmm.n_components):
            mu = gmm.means[k]
            ax.scatter(mu[0], mu[1], s=80*gmm.weights[k]+30,
                       color="white", edgecolors=col, linewidths=2, zorder=5)
            ax.annotate(f"w={gmm.weights[k]:.2f}", mu,
                        xytext=(4, 4), textcoords="offset points", fontsize=7)
        ax.set_title(name, fontsize=9, fontweight="bold", color=col)
        ax.set_xlabel("q1");  ax.set_ylabel("q2")
        ax.set_xlim(x0, x1);  ax.set_ylim(y0, y1)
        ax.set_aspect("equal")
        plt.colorbar(im, ax=ax, shrink=0.75, label="Densite")


def plot_distributions(exp: dict, save_dir: Path):
    """Genere distributions.png pour une experience."""
    gmm0, gmm1 = exp["gmm0"], exp["gmm1"]
    d = gmm0.dim

    if d == 1:
        fig, ax = plt.subplots(figsize=(8, 4))
        _plot_1d_ax(ax, gmm0, gmm1, exp["label"], exp.get("ref"))
    else:
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(exp["label"], fontsize=10, fontweight="bold")
        _plot_2d_pair(ax0, ax1, gmm0, gmm1)

    plt.tight_layout()
    path = save_dir / "distributions.png"
    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  distributions.png sauvegarde")


def plot_distributions_overview(experiments: list, save_path: Path):
    """Figure recapitulative : toutes les experiences sur une image."""
    n   = len(experiments)
    heights = [1.0 if e["gmm0"].dim == 1 else 1.6 for e in experiments]

    fig = plt.figure(figsize=(14, sum(heights) * 3.0))
    fig.suptitle("Distributions de depart (bleu) et d'arrivee (rouge)",
                 fontsize=13, fontweight="bold", y=1.01)

    gs = gridspec.GridSpec(n, 2, figure=fig,
                           height_ratios=heights, hspace=0.6, wspace=0.35)

    for i, exp in enumerate(experiments):
        gmm0, gmm1 = exp["gmm0"], exp["gmm1"]
        if gmm0.dim == 1:
            ax = fig.add_subplot(gs[i, :])
            _plot_1d_ax(ax, gmm0, gmm1, exp["label"], exp.get("ref"))
        else:
            ax0 = fig.add_subplot(gs[i, 0])
            ax1 = fig.add_subplot(gs[i, 1])
            # Titre de ligne
            ypos = ax0.get_position().y1
            fig.text(0.5, ypos + 0.005, exp["label"],
                     ha="center", fontsize=9, fontweight="bold",
                     transform=fig.transFigure)
            _plot_2d_pair(ax0, ax1, gmm0, gmm1)

    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\ndistributions_overview.png sauvegarde -> {save_path}")


# =============================================================================
# REFERENCE PAR MONTE-CARLO
# =============================================================================

def compute_reference(gmm0, gmm1, n=1_000_000, seed=0):
    from scipy.special import logsumexp
    rng = np.random.default_rng(seed)
    batch = 50_000
    log_ratios = []
    for _ in range(n // batch):
        qs = gmm0.sample(batch, rng=rng)
        lr = np.array([
            gmm1.log_density(qs[i]) - gmm0.log_density(qs[i])
            for i in range(batch)
        ])
        log_ratios.append(lr)
    log_ratio = np.concatenate(log_ratios)
    return float(np.exp(logsumexp(log_ratio) - np.log(n)))


# =============================================================================
# CHARGEMENT OU ENTRAINEMENT
# =============================================================================

def load_or_train(name, train_fn, cfg, kwargs, out_dir):
    params_path = out_dir / f"{name}_params.pkl"
    meta_path   = out_dir / f"{name}_meta.pkl"

    if params_path.exists() and meta_path.exists():
        print(f"  Chargement {name.upper()} depuis {params_path.name}")
        with open(params_path, "rb") as f:
            params = pickle.load(f)
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        return params, [], meta

    print(f"  Entrainement {name.upper()} ...")
    t0 = time.time()
    params, losses, meta = train_fn(cfg, **kwargs)
    print(f"  Termine en {time.time()-t0:.0f}s")

    with open(params_path, "wb") as f:
        pickle.dump(params, f)
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)
    return params, losses, meta


# =============================================================================
# PIPELINE D'UNE EXPERIENCE
# =============================================================================

def run_experiment(exp: dict) -> dict:
    name  = exp["name"]
    label = exp["label"]
    gmm0  = exp["gmm0"]
    gmm1  = exp["gmm1"]
    T     = exp["T"]

    out_dir = BASE_OUTPUT / name
    out_dir.mkdir(exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"  d={gmm0.dim}  K0={gmm0.n_components}  K1={gmm1.n_components}  T={T}")
    print(f"{'='*70}")

    # -- 1. Visualisation des distributions -----------------------------------
    print("  Visualisation des distributions...")
    plot_distributions(exp, out_dir)

    # -- 2. Reference ---------------------------------------------------------
    ref = exp.get("ref")
    if ref is None:
        print("  Calcul reference MC (1M samples)...")
        ref = compute_reference(gmm0, gmm1)
    ref_dF = float(-np.log(max(ref, 1e-300)))
    print(f"  Reference : Z1/Z0 = {ref:.5f}   dF = {ref_dF:.5f}")

    # -- 3. Configs -----------------------------------------------------------
    cfg_train = PipelineConfig(
        gmm0=gmm0, gmm1=gmm1,
        T=T, dt=T/K_TRAIN,
        schedule=np.linspace(0, 1, K_TRAIN + 1),
        n_samples=BATCH_SIZE, seed=SEED,
    )
    N_steps_eval = int(T / DT_EVAL)
    cfg_eval = PipelineConfig(
        gmm0=gmm0, gmm1=gmm1,
        T=T, dt=DT_EVAL,
        schedule=np.linspace(0, 1, N_steps_eval + 1),
        n_samples=N_SAMPLES, seed=SEED + 1,
    )

    base = dict(n_epochs=N_EPOCHS_MAX, early_stop_patience=EARLY_STOP_PAT,
                batch_size=BATCH_SIZE, emb_dim=EMB_DIM)

    mcd_kw  = dict(**base, lr_init=1e-3, lr_peak=5e-3, lr_end=5e-5, K_mcd=K_TRAIN)
    cmcd_kw = dict(**base, lr_init=5e-4, lr_peak=3e-3, lr_end=1e-5,
                   grad_clip=0.5, K_cmcd=K_TRAIN)
    led_kw  = dict(**base, lr_init=1e-3, lr_peak=5e-3, lr_end=5e-5,
                   exact_div=False, n_probes=8, grad_clip=1.0,
                   loss_type="log_variance", K_led=K_TRAIN)

    # -- 4. Entrainement ------------------------------------------------------
    print("\n-- Entrainement ------------------------------------------------")
    mcd_p,  mcd_l,  mcd_m  = load_or_train("mcd",  gmm_mcd.train,  cfg_train, mcd_kw,  out_dir)
    cmcd_p, cmcd_l, cmcd_m = load_or_train("cmcd", gmm_cmcd.train, cfg_train, cmcd_kw, out_dir)
    led_p,  led_l,  led_m  = load_or_train("led",  gmm_led.train,  cfg_train, led_kw,  out_dir)

    # -- 5. Evaluation --------------------------------------------------------
    print(f"\n-- Evaluation (dt={DT_EVAL}, K={N_steps_eval}, n={N_SAMPLES}) --")
    results = {
        "AIS":  gmm_ais.run(cfg_eval, eval_batch_size=EVAL_BATCH_SIZE),
        "MCD":  gmm_mcd.estimate(cfg_eval,  mcd_p,  mcd_m,
                                  emb_dim=EMB_DIM, eval_batch_size=EVAL_BATCH_SIZE),
        "CMCD": gmm_cmcd.estimate(cfg_eval, cmcd_p, cmcd_m,
                                   emb_dim=EMB_DIM, eval_batch_size=EVAL_BATCH_SIZE),
        "LED":  gmm_led.estimate(cfg_eval,  led_p,  led_m,
                                  emb_dim=EMB_DIM, eval_batch_size=EVAL_BATCH_SIZE),
    }

    # -- 6. Metriques ---------------------------------------------------------
    print("\n-- Metriques ---------------------------------------------------")
    all_metrics = print_comparison_table(results, reference_ratio=ref)
    variance_reduction_table(all_metrics, baseline="AIS")

    with open(out_dir / "metrics.txt", "w", encoding="utf-8") as f:
        f.write(f"Experience : {label}\n")
        f.write(f"Reference  : Z1/Z0 = {ref:.6f}   dF = {ref_dF:.6f}\n\n")
        for algo, m in all_metrics.items():
            f.write(f"{algo}:\n")
            f.write(f"  ratio_estimate = {m['ratio_estimate']:.6f}\n")
            f.write(f"  dF_estimate    = {m['dF_estimate']:.6f}\n")
            f.write(f"  ratio_variance = {m['ratio_variance']:.4e}\n")
            f.write(f"  work_variance  = {m['work_variance']:.4e}\n")
            f.write(f"  n_nan          = {m.get('n_nan', 0)}\n\n")

    # -- 7. Plots -------------------------------------------------------------
    plot_work_histograms(results, reference_dF=ref_dF,
                         save_path=out_dir / "work_histograms.png",
                         title=f"Work histograms — {label}")
    plot_ratio_convergence(results, reference_ratio=ref,
                           save_path=out_dir / "ratio_convergence.png")
    loss_histories = {k: v for k, v in
                      [("MCD", mcd_l), ("CMCD", cmcd_l), ("LED", led_l)]
                      if len(v) > 0}
    if loss_histories:
        plot_loss_curves(loss_histories, reference_dF=ref_dF,
                         save_path=out_dir / "loss_curves.png")

    print(f"\n  OK — resultats dans {out_dir}")
    return all_metrics


# =============================================================================
# BOUCLE PRINCIPALE
# =============================================================================

def main():
    start_time = time.time()
    print(f"\n{'#'*70}")
    print(f"  EXPERIENCES — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  {len(EXPERIMENTS)} experiences planifiees")
    print(f"{'#'*70}")

    # Vue d'ensemble des distributions AVANT de commencer
    print("\nGeneration de la vue d'ensemble des distributions...")
    plot_distributions_overview(EXPERIMENTS, BASE_OUTPUT / "distributions_overview.png")

    summary = {}

    for i, exp in enumerate(EXPERIMENTS, 1):
        print(f"\n\n{'#'*70}")
        print(f"  EXPERIENCE {i}/{len(EXPERIMENTS)} : {exp['name']}")
        print(f"{'#'*70}")
        t0 = time.time()
        try:
            metrics = run_experiment(exp)
            summary[exp["name"]] = {"label": exp["label"],
                                    "metrics": metrics, "status": "OK"}
        except Exception as e:
            print(f"\n  ERREUR dans {exp['name']} :")
            traceback.print_exc()
            summary[exp["name"]] = {"label": exp["label"], "metrics": None,
                                    "status": f"FAILED: {e}"}

        print(f"\n  Experience {i} terminee en {(time.time()-t0)/60:.1f} min")

    # -- Rapport final --------------------------------------------------------
    total_time = time.time() - start_time
    print(f"\n\n{'#'*70}")
    print(f"  RESUME — duree totale : {total_time/60:.0f} min")
    print(f"{'#'*70}\n")

    lines = [
        f"Rapport — {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Duree totale : {total_time/60:.0f} min",
        "=" * 70, "",
    ]
    for exp_name, s in summary.items():
        lines.append(f"\n{s['label']}")
        lines.append(f"Status : {s['status']}")
        if s["metrics"] is not None:
            m = s["metrics"]
            lines.append(f"{'Algo':<10} {'Ratio est.':>12} {'dF est.':>10} "
                         f"{'Var(W)':>14} {'VR(W)/AIS':>12}")
            lines.append("-" * 62)
            ais_vw = m.get("AIS", {}).get("work_variance", 1.0)
            for algo, met in m.items():
                vw = met.get("work_variance", float("nan"))
                vr = ais_vw / vw if vw > 0 else float("nan")
                lines.append(
                    f"{algo:<10} {met.get('ratio_estimate', float('nan')):>12.5f} "
                    f"{met.get('dF_estimate', float('nan')):>10.4f} "
                    f"{vw:>14.4e} {vr:>12.1f}"
                )
        lines.append("")

    report = "\n".join(lines)
    print(report)
    with open(BASE_OUTPUT / "summary.txt", "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Rapport sauvegarde -> {BASE_OUTPUT / 'summary.txt'}")


if __name__ == "__main__":
    main()
