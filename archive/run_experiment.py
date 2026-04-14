"""
run_experiment.py
─────────────────
Compare AIS / MCD / CMCD / LED sur un problème GMM.

Vitesse :
  - Entraînement K=64 chaînes courtes  → rapide, évite OOM
  - Évaluation en mini-batches de 256  → mémoire bornée
  - DT_EVAL=1e-3 (même que training)   → évaluation rapide
    Pour plus de précision, utiliser DT_EVAL=1e-4 mais ça prend plus de temps
  - N_SAMPLES=10_000 trajectoires       → statistiques robustes

Premier run : entraîne et sauvegarde params + meta (~5-15 min).
Runs suivants : charge depuis disque (<1 min) puis évalue.
Pour ré-entraîner : supprimer les .pkl ou FORCE_RETRAIN = True.
"""

import pickle
import numpy as np
from pathlib import Path

from gmm_config import GMMParams, PipelineConfig
import gmm_ais, gmm_mcd, gmm_cmcd, gmm_led
from gmm_metrics import print_comparison_table, variance_reduction_table
from gmm_plots import plot_work_histograms, plot_loss_curves, plot_ratio_convergence


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

GMM0 = GMMParams.single_gaussian(mean=[-2.0], cov=[[1.0]])
GMM1 = GMMParams.single_gaussian(mean=[ 2.0], cov=[[0.25]])

T = 1.0

# Évaluation
DT_EVAL          = 1e-3   # même que training → très rapide
                           # mettre 1e-4 pour plus de précision (10x plus lent)
N_SAMPLES        = 10_000
EVAL_BATCH_SIZE  = 256    # trajectoires par mini-batch — ajuster selon RAM
SEED             = 42

# Entraînement — K=64 pour tous les algos NN
K_TRAIN        = 128
N_EPOCHS_MAX   = 5000
EARLY_STOP_PAT = 500
BATCH_SIZE     = 128
EMB_DIM        = 20

FORCE_RETRAIN   = False
REFERENCE_RATIO = 0.5

OUTPUT_DIR = Path("./results")
OUTPUT_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def load_or_train(name, train_fn, cfg, kwargs):
    params_path = OUTPUT_DIR / f"{name}_params.pkl"
    meta_path   = OUTPUT_DIR / f"{name}_meta.pkl"

    if params_path.exists() and meta_path.exists() and not FORCE_RETRAIN:
        print(f"  Chargement {name.upper()} depuis {params_path}")
        with open(params_path, "rb") as f:
            params = pickle.load(f)
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        return params, [], meta

    print(f"  Entraînement {name.upper()} ...")
    params, losses, meta = train_fn(cfg, **kwargs)
    with open(params_path, "wb") as f:
        pickle.dump(params, f)
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)
    print(f"  Sauvegardé → {params_path}")
    return params, losses, meta


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 1 — Configs
# ─────────────────────────────────────────────────────────────────────────────

cfg_train = PipelineConfig(
    gmm0=GMM0, gmm1=GMM1,
    T=T, dt=T/K_TRAIN,
    schedule=np.linspace(0, 1, K_TRAIN + 1),
    n_samples=BATCH_SIZE,
    seed=SEED,
)

N_STEPS_EVAL = int(T / DT_EVAL)
cfg_eval = PipelineConfig(
    gmm0=GMM0, gmm1=GMM1,
    T=T, dt=DT_EVAL,
    schedule=np.linspace(0, 1, N_STEPS_EVAL + 1),
    n_samples=N_SAMPLES,
    seed=SEED + 1,
)

base = dict(
    n_epochs=N_EPOCHS_MAX,
    early_stop_patience=EARLY_STOP_PAT,
    batch_size=BATCH_SIZE,
    emb_dim=EMB_DIM,
)

"""
mcd_kwargs  = dict(**base, lr_init=1e-3,  lr_peak=5e-3, lr_end=5e-5,  K_mcd=K_TRAIN)
cmcd_kwargs = dict(**base, lr_init=2e-4,  lr_peak=1e-3, lr_end=1e-5,
                   grad_clip=0.5, K_cmcd=K_TRAIN)
led_kwargs  = dict(**base, lr_init=1e-3,  lr_peak=5e-3, lr_end=5e-5,
                   exact_div=False, n_probes=4, grad_clip=1.0,
                   loss_type="work_variance", K_led=K_TRAIN)

"""


mcd_kwargs = dict(**base,
    lr_init=1e-3, lr_peak=5e-3, lr_end=5e-5,
    K_mcd=K_TRAIN,
)

cmcd_kwargs = dict(**base,
    lr_init=5e-4, lr_peak=3e-3, lr_end=1e-5,   # LR augmenté
    grad_clip=0.3,                               # clip réduit
    K_cmcd=K_TRAIN,
)

led_kwargs = dict(**base,
    lr_init=1e-3, lr_peak=5e-3, lr_end=5e-5,
    exact_div=False, n_probes=8,                 # plus de probes
    grad_clip=1.0,
    loss_type="work_mean",                       # changer ici
    K_led=K_TRAIN,
)

# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 2 — Entraînement
# ─────────────────────────────────────────────────────────────────────────────

print("\n── MCD ─────────────────────────────────────────────────────────────")
mcd_params,  mcd_losses,  mcd_meta  = load_or_train("mcd",  gmm_mcd.train,  cfg_train, mcd_kwargs)

print("\n── CMCD ────────────────────────────────────────────────────────────")
cmcd_params, cmcd_losses, cmcd_meta = load_or_train("cmcd", gmm_cmcd.train, cfg_train, cmcd_kwargs)

print("\n── LED ─────────────────────────────────────────────────────────────")
led_params,  led_losses,  led_meta  = load_or_train("led",  gmm_led.train,  cfg_train, led_kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 3 — Évaluation
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n── Évaluation (dt={DT_EVAL}, K={N_STEPS_EVAL}, "
      f"n={N_SAMPLES}, batch={EVAL_BATCH_SIZE}) ────")

results = {
    "AIS":  gmm_ais.run(cfg_eval,
                        eval_batch_size=EVAL_BATCH_SIZE),
    "MCD":  gmm_mcd.estimate(cfg_eval,  mcd_params,  mcd_meta,
                              emb_dim=EMB_DIM, eval_batch_size=EVAL_BATCH_SIZE),
    "CMCD": gmm_cmcd.estimate(cfg_eval, cmcd_params, cmcd_meta,
                               emb_dim=EMB_DIM, eval_batch_size=EVAL_BATCH_SIZE),
    "LED":  gmm_led.estimate(cfg_eval,  led_params,  led_meta,
                              emb_dim=EMB_DIM, eval_batch_size=EVAL_BATCH_SIZE),
}


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 4 — Métriques
# ─────────────────────────────────────────────────────────────────────────────

print("\n── Métriques ───────────────────────────────────────────────────────")
all_metrics = print_comparison_table(results, reference_ratio=REFERENCE_RATIO)
variance_reduction_table(all_metrics, baseline="AIS")


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 5 — Plots
# ─────────────────────────────────────────────────────────────────────────────

ref_dF = float(-np.log(REFERENCE_RATIO))

plot_work_histograms(results, reference_dF=ref_dF,
                     save_path=OUTPUT_DIR / "work_histograms.png")
plot_ratio_convergence(results, reference_ratio=REFERENCE_RATIO,
                       save_path=OUTPUT_DIR / "ratio_convergence.png")

loss_histories = {k: v for k, v in
                  [("MCD", mcd_losses), ("CMCD", cmcd_losses), ("LED", led_losses)]
                  if len(v) > 0}
if loss_histories:
    plot_loss_curves(loss_histories, reference_dF=ref_dF,
                     save_path=OUTPUT_DIR / "loss_curves.png")

print(f"\nTerminé. Résultats dans {OUTPUT_DIR.resolve()}")
