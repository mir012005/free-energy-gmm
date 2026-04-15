"""
main.py
───────
Évalue les modèles, calcule les métriques (tableaux + textes)
et trace une figure complète (Histogrammes séparés, Trajectoires, Densités, Convergence).
"""
import dataclasses
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import os
import math
import jax
import jax.numpy as jnp
from datetime import datetime

from physics import GMMParams, PipelineConfig
from algorithms import run_ais, train_and_eval_mcd, train_and_eval_cmcd, train_and_eval_led

# 1. OUTILS D'ANALYSE ET MÉTRIQUES
def compute_metrics(works, name):
    w = np.asarray(works, dtype=float)
    w_finite = w[np.isfinite(w)] 
    
    if len(w_finite) < 2:
        return {"name": name, "ratio": np.nan, "var_ratio": np.nan, "dF": np.nan, "var_dF": np.nan, "var_W": np.nan}
        
    e_mW = np.exp(-w_finite)
    ratio = float(e_mW.mean())
    var_ratio = float(e_mW.var(ddof=1))
    dF = float(-np.log(ratio + 1e-300))

    # La vraie variance de l'estimateur nécessite de diviser par N
    N = len(w_finite)
    var_dF = (var_ratio / N) / (ratio**2 + 1e-300)
    #var_dF = var_ratio / (ratio**2 + 1e-300)

    var_W = float(w_finite.var(ddof=1))
    
    return {"name": name, "ratio": ratio, "var_ratio": var_ratio, "dF": dF, "var_dF": var_dF, "var_W": var_W}

def print_table(results_metrics, output_dir):
    lines = ["="*100]
    lines.append(f"{'Algorithm':<15} | {'Ratio Est.':<13} | {'Ratio Var.':<13} | {'dF Estimate':<13} | {'dF Variance':<13} | {'Work Var.'}")
    lines.append("-" * 100)
    
    for m in results_metrics:
        ratio_str = f"{m['ratio']:.3e}"
        var_r_str = f"{m['var_ratio']:.3e}" if m['var_ratio'] > 1e-10 else "0"
        df_str = f"{m['dF']:.3f}" 
        var_df_str = f"{m['var_dF']:.3e}" if m['var_dF'] > 1e-10 else "0"
        var_w_str = f"{m['var_W']:.2f}" if m['var_W'] > 1 else (f"{m['var_W']:.3e}" if m['var_W'] >= 1e-10 else "0")
            
        lines.append(f"{m['name']:<15} | {ratio_str:<13} | {var_r_str:<13} | {df_str:<13} | {var_df_str:<13} | {var_w_str:<13}")
        
    lines.append("="*100)
    output_text = "\n".join(lines)

    print("\n" + output_text + "\n")
    filepath = os.path.join(output_dir, "resultats_experience.txt") # <-- Sauvegarde ciblée
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("=== TABLEAU DES PERFORMANCES ===\n" + output_text + "\n")

def get_running_estimators(works):
    w = np.asarray(works, dtype=float)[np.isfinite(np.asarray(works, dtype=float))]
    cum_mean = np.cumsum(np.exp(-w)) / np.arange(1, len(w) + 1)
    return -np.log(cum_mean + 1e-300)

def pdf_gmm(q_grid, gmm: GMMParams):
    return np.sum([w * scipy.stats.norm.pdf(q_grid, loc=m[0], scale=np.sqrt(c[0,0])) for w, m, c in zip(gmm.weights, gmm.means, gmm.covs)], axis=0)


# 2. MODULE D'AFFICHAGE
def generate_all_plots(algos_data, true_dF, gmm1, n_samples, output_dir):
    plt.rcParams.update({'font.size': 11})
    DPI = 300

    # Extraire les données spécifiques pour la lisibilité
    w_ais, trj_ais = algos_data["AIS"]["works"], algos_data["AIS"]["trajs"]
    w_led, trj_led = algos_data["LED"]["works"], algos_data["LED"]["trajs"]

    # FIG 1: Histogrammes W
    fig1, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig1.suptitle("Distribution des travaux virtuels (W)", fontsize=14, fontweight='bold')
    for ax, (name, data) in zip(axes.flatten(), algos_data.items()):
        w_clean = data["works"][np.isfinite(data["works"])]
        w_filtered = w_clean[(w_clean >= np.percentile(w_clean, 0.1)) & (w_clean <= np.percentile(w_clean, 99.9))]
        ax.hist(w_filtered, bins=50, density=True, color=data["color"], alpha=0.8)
        ax.axvline(true_dF, color='red', linestyle='dashed', linewidth=2, label=r"Vrai $\Delta F$")
        ax.axvline(-np.log(np.mean(np.exp(-w_clean)) + 1e-300), color='black', linewidth=2, label=r"$\Delta F$ estimé")
        ax.set_title(name)
        if name == "AIS": ax.legend()
    fig1.tight_layout()
    fig1.savefig(os.path.join(output_dir, "fig1_histogrammes_W.png"), dpi=DPI, bbox_inches='tight')
    plt.close(fig1)

    # FIG 2: Trajectoires
    fig2 = plt.figure(figsize=(10, 5))
    plt.plot(np.linspace(0, 1.0, trj_ais.shape[1]), trj_ais[:50, :, 0].T, color="gray", alpha=0.2)
    plt.plot(np.linspace(0, 1.0, trj_led.shape[1]), trj_led[:50, :, 0].T, color="green", alpha=0.3)
    plt.plot([], [], color="gray", label="AIS"), plt.plot([], [], color="green", label="LED")
    plt.title("Dynamique des trajectoires (50 particules)", fontweight='bold')
    plt.legend()
    fig2.tight_layout()
    fig2.savefig(os.path.join(output_dir, "fig2_trajectoires.png"), dpi=DPI, bbox_inches='tight')
    plt.close(fig2)

    # FIG 3: Densité Finale
    fig3 = plt.figure(figsize=(10, 5))
    q_grid = np.linspace(-4, 4, 300)
    plt.hist(trj_ais[:, -1, 0], bins=50, density=True, alpha=0.4, color="gray", label="AIS Final")
    plt.hist(trj_led[:, -1, 0], bins=50, density=True, alpha=0.5, color="green", label="LED Final")
    plt.plot(q_grid, pdf_gmm(q_grid, gmm1), 'k--', linewidth=2, label=r"Cible $\pi_1$")
    plt.title("Distribution spatiale à t=1", fontweight='bold')
    plt.legend()
    fig3.tight_layout()
    fig3.savefig(os.path.join(output_dir, "fig3_densite_finale.png"), dpi=DPI, bbox_inches='tight')
    plt.close(fig3)

    # FIG 4: Convergence
    fig4 = plt.figure(figsize=(10, 5))
    for name, data in algos_data.items():
        est = get_running_estimators(data["works"])
        plt.plot(np.arange(1, len(est) + 1), est, color=data["color"], label=name, alpha=0.8) # <-- Adaptatif !
    """
    fig4 = plt.figure(figsize=(10, 5))
    for name, data in algos_data.items():
        plt.plot(np.arange(1, n_samples + 1), get_running_estimators(data["works"]), color=data["color"], label=name, alpha=0.8)
    """

    plt.axhline(true_dF, color='k', linestyle='dashed', linewidth=2, label="Cible théorique")
    plt.title("Convergence de l'estimateur d'Énergie Libre", fontweight='bold')
    plt.ylim(true_dF - 0.5, true_dF + 1.0)
    plt.legend()
    fig4.tight_layout()
    fig4.savefig(os.path.join(output_dir, "fig4_convergence.png"), dpi=DPI, bbox_inches='tight')
    plt.close(fig4)
    
    # --- FIGURE 5 : Évolution de la Loss ---
    fig5 = plt.figure(figsize=(10, 5))
    
    for name, data in algos_data.items():
        if data.get("losses") is not None:
            # La loss est sauvegardée toutes les 50 époques dans _train_loop
            epochs = np.arange(len(data["losses"])) * 50
            plt.plot(epochs, data["losses"], color=data["color"], label=name, linewidth=2)

    plt.title("Évolution de la fonction de perte (Loss) durant l'entraînement", fontweight='bold')
    plt.xlabel("Époques")
    plt.ylabel("Loss (ELBO ou Var(W))")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    fig5.tight_layout()
    fig5.savefig(os.path.join(output_dir, "fig5_loss_evolution.png"), dpi=DPI, bbox_inches='tight')
    plt.close(fig5)
    
    print(f"✅ Figures générées avec succès dans : {output_dir}")

# Sauvegarde les hyperparamètres de chaque algorithme dans un fichier texte lisible.
def save_configurations(configs, filepath):
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("=== HYPERPARAMÈTRES DE L'EXPÉRIENCE ===\n\n")
        for algo_name, cfg in configs.items():
            f.write(f"--- CONFIGURATION {algo_name} ---\n")
            # On convertit le dataclass en dictionnaire
            for key, value in dataclasses.asdict(cfg).items():
                # On évite d'imprimer les tableaux de 10 000 lignes du schedule
                if hasattr(value, "shape") and np.prod(value.shape) > 10:
                    value = f"Array de dimension {value.shape}"
                f.write(f"{key} : {value}\n")
            f.write("\n")

# 3. EXÉCUTION PRINCIPALE
def main():
    # 0. Création du dossier d'archivage unique pour ce run
    timestamp = datetime.now().strftime("%Y-%m-%d_%Hh%Mm%S")
    #run_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", f"run_{timestamp}")
    base_dir = os.getcwd()
    run_dir = os.path.join(base_dir, "runs", f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    print(f"\n📂 Dossier de sauvegarde créé : {run_dir}")

    # 1. Configuration (Paramètres stricts du professeur)
    cfg_base = PipelineConfig(
        gmm0=GMMParams.single_gaussian(mean=[-2.0], cov=[[1.0]]),
        gmm1=GMMParams.single_gaussian(mean=[ 2.0], cov=[[0.25]]),
        T=1.0, seed=42, n_samples=10000, dt_train=1e-3, dt_eval=1e-4,
        batch_size_train=128, batch_size_val=10000, emb_dim=20,
        clip_norm=float('inf'), weight_decay=0.0, patience=999999,
        lr_init=0.005
    )
    
    configs = {
        "AIS": cfg_base,
        "MCD": dataclasses.replace(cfg_base, n_epochs=5000, dt_eval=1e-3),
        "CMCD": dataclasses.replace(cfg_base, n_epochs=5000),
        "LED": dataclasses.replace(cfg_base, n_epochs=500)
    }

    save_configurations(configs, os.path.join(run_dir, "configuration.txt"))

    # 2. Exécution des algorithmes
    print("\n--- Démarrage des simulations ---")
    w_ais, trj_ais = run_ais(configs["AIS"])
    (w_mcd, trj_mcd), loss_mcd   = train_and_eval_mcd(configs["MCD"])
    (w_cmcd, trj_cmcd), loss_cmcd = train_and_eval_cmcd(configs["CMCD"])
    (w_led, trj_led), loss_led   = train_and_eval_led(configs["LED"])

    # 3. Structuration des données
    algos_data = {
        # AIS n'a pas d'entraînement, on lui met None
        "AIS":  {"works": w_ais,  "trajs": trj_ais,  "color": "gray",   "losses": None},
        "MCD":  {"works": w_mcd,  "trajs": trj_mcd,  "color": "blue",   "losses": loss_mcd},
        "CMCD": {"works": w_cmcd, "trajs": trj_cmcd, "color": "orange", "losses": loss_cmcd},
        "LED":  {"works": w_led,  "trajs": trj_led,  "color": "green",  "losses": loss_led}
    }

    # 4. Calcul et affichage des métriques
    metrics = [compute_metrics(data["works"], name) for name, data in algos_data.items()]
    print_table(metrics, run_dir)

    # 5. Génération des graphiques
    ref_ratio = 0.5
    true_dF = -np.log(ref_ratio)
    generate_all_plots(algos_data, true_dF, cfg_base.gmm1, cfg_base.n_samples, run_dir)

if __name__ == "__main__":
    main()