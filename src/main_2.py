"""
main_2.py (Version Corrigée - Ring of Gaussians 2D)
─────────
Visualisation avec double ligne : 
- Rouge (Pointillés) : Vrai ΔF théorique (1.609)
- Noire (Pleine) : ΔF estimé par l'algorithme
"""
import dataclasses
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import os
import math
import jax.numpy as jnp
from datetime import datetime

# Tes modules personnalisés
from physics import GMMParams, PipelineConfig
from algorithms import run_ais, train_and_eval_mcd, train_and_eval_cmcd, train_and_eval_led

# ==================================================================================================
# 1. OUTILS D'ANALYSE
# ==================================================================================================

def compute_metrics(works, name):
    w = np.asarray(works, dtype=float)
    w_finite = w[np.isfinite(w)] 
    
    if len(w_finite) < 2:
        return {"name": name, "ratio": np.nan, "dF": np.nan, "var_W": np.nan}
        
    e_mW = np.exp(-w_finite)
    ratio = float(e_mW.mean())
    dF = float(-np.log(ratio + 1e-300))
    var_W = float(w_finite.var(ddof=1))
    
    return {"name": name, "ratio": ratio, "dF": dF, "var_W": var_W}

def print_table(results_metrics, output_dir):
    lines = ["="*100]
    lines.append(f"{'Algorithm':<15} | {'Ratio Est.':<13} | {'dF Estimate':<13} | {'Work Var.'}")
    lines.append("-" * 100)
    for m in results_metrics:
        lines.append(f"{m['name']:<15} | {m['ratio']:.3e} | {m['dF']:.3f} | {m['var_W']:.3e}")
    lines.append("="*100)
    output_text = "\n".join(lines)
    print("\n" + output_text + "\n")
    with open(os.path.join(output_dir, "resultats_2D.txt"), "w") as f:
        f.write(output_text)

def get_running_estimators(works):
    w = np.asarray(works, dtype=float)[np.isfinite(np.asarray(works, dtype=float))]
    cum_mean = np.cumsum(np.exp(-w)) / np.arange(1, len(w) + 1)
    return -np.log(cum_mean + 1e-300)

# ==================================================================================================
# 2. MODULE D'AFFICHAGE 2D MIS À JOUR
# ==================================================================================================

def generate_all_plots(algos_data, true_dF, gmm1, n_samples, output_dir):
    plt.rcParams.update({'font.size': 12, 'axes.labelsize': 11})
    DPI = 300
    
    # ---------------------------------------------------------
    # FIG 1: Histogrammes W (Avec axes standardisés pour l'IA)
    # ---------------------------------------------------------
    fig1, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig1.suptitle("Distribution du Travail Virtuel (W)", fontsize=16, fontweight='bold')
    
    for ax, (name, data) in zip(axes.flatten(), algos_data.items()):
        w_clean = data["works"][np.isfinite(data["works"])]
        est_dF = -np.log(np.mean(np.exp(-w_clean)) + 1e-300)
        
        # On calcule dynamiquement le nombre de barres pour que chaque barre ait la même finesse
        bins = np.linspace(min(w_clean), max(w_clean), 200)
        ax.hist(w_clean, bins=bins, density=True, color=data["color"], alpha=0.7, edgecolor='white', linewidth=0.5)
        ax.axvline(true_dF, color='red', linestyle='--', linewidth=2, label=f"Vrai $\Delta F$ ({true_dF:.2f})")
        ax.axvline(est_dF, color='black', linestyle='-', linewidth=2, label=f"Estimé ({est_dF:.2f})")
        
        ax.set_title(name, fontweight='bold')
        ax.set_xlabel("Travail Virtuel W")
        ax.set_ylabel("Densité")
        
        # On force la même échelle X pour MCD, CMCD et LED pour comparer la variance
        if name in ["MCD", "CMCD", "LED"]:
            ax.set_xlim([true_dF - 4, true_dF + 7])
            
        if name == "AIS": 
            ax.legend(loc="upper right", fontsize=10)
            
    fig1.tight_layout()
    fig1.savefig(os.path.join(output_dir, "fig1_hist_W.png"), dpi=DPI, bbox_inches='tight')
    plt.close(fig1)

    # ---------------------------------------------------------
    # FIG 2: Trajectoires 2D (Séparées côte à côte)
    # ---------------------------------------------------------
    trj_ais = algos_data["AIS"]["trajs"]
    trj_led = algos_data["LED"]["trajs"]
    target_means = np.array(gmm1.means)
    
    fig2, axes = plt.subplots(1, 2, figsize=(14, 7), sharex=True, sharey=True)
    fig2.suptitle("Dynamique des trajectoires : AIS vs LED (50 particules)", fontsize=16, fontweight='bold')
    
    for ax, trj, color, title in zip(axes, [trj_ais, trj_led], ["gray", "green"], ["AIS (Standard)", "LED (Assisté par IA)"]):
        for i in range(50): 
            ax.plot(trj[i, :, 0], trj[i, :, 1], color=color, alpha=0.2, lw=0.8)
        ax.scatter(target_means[:, 0], target_means[:, 1], color="black", marker="x", s=80, linewidth=2, zorder=5)
        ax.scatter([0], [0], color="black", marker="o", s=100, zorder=10, label="Point de départ $\pi_0$")
        ax.legend()
        ax.set_title(title)
        ax.set_xlabel("Position X")
        ax.set_ylabel("Position Y")
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_aspect('equal')
    fig2.tight_layout()
    fig2.savefig(os.path.join(output_dir, "fig2_trajectoires_2D.png"), dpi=DPI, bbox_inches='tight')
    plt.close(fig2)

    # ---------------------------------------------------------
    # FIG 3: Densité Finale (Séparée avec cibles)
    # ---------------------------------------------------------
    fig3, axes = plt.subplots(1, 2, figsize=(14, 7), sharex=True, sharey=True)
    fig3.suptitle("Distribution spatiale finale à t=1", fontsize=16, fontweight='bold')
    
    for ax, trj, color, title in zip(axes, [trj_ais, trj_led], ["gray", "green"], ["AIS", "LED"]):
        ax.scatter(trj[:, -1, 0], trj[:, -1, 1], color=color, alpha=0.3, s=4, edgecolors='none')
        ax.scatter(target_means[:, 0], target_means[:, 1], color="black", marker="x", s=80, linewidth=2, label="Centres $\pi_1$")
        
        # Ajout des cercles pour illustrer l'écart-type des cibles
        for m in target_means:
            circle = plt.Circle((m[0], m[1]), radius=np.sqrt(0.2), color='black', fill=False, linestyle=':', alpha=0.5)
            ax.add_patch(circle)
            
        ax.set_title(title)
        ax.set_xlabel("Position X")
        ax.set_ylabel("Position Y")
        ax.set_aspect('equal')
        if title == "AIS": ax.legend()
        
    fig3.tight_layout()
    fig3.savefig(os.path.join(output_dir, "fig3_scatter_2D.png"), dpi=DPI, bbox_inches='tight')
    plt.close(fig3)

    # ---------------------------------------------------------
    # FIG 4: Évolution de la Loss
    # ---------------------------------------------------------
    fig4 = plt.figure(figsize=(10, 6))
    for name, data in algos_data.items():
        if data["losses"] is not None:
            epochs = np.arange(len(data["losses"])) * 50
            plt.plot(epochs, data["losses"], color=data["color"], label=name, lw=2)
            
    # La ligne de la vérité thermodynamique
    plt.axhline(true_dF, color='red', linestyle='--', lw=2, label=f"Borne minimale (Vrai $\Delta F$ = {true_dF:.2f})")
    
    plt.title("Convergence de la fonction de Perte (Travail Moyen)", fontsize=14, fontweight='bold')
    plt.xlabel("Époques")
    plt.ylabel("Loss")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=11)
    
    # On ajuste l'axe Y pour bien voir l'écrasement sur la ligne rouge
    plt.ylim([0, true_dF + 4.0]) 
    
    fig4.tight_layout()
    fig4.savefig(os.path.join(output_dir, "fig4_loss.png"), dpi=DPI, bbox_inches='tight')
    plt.close(fig4)

    print(f"✅ Figures Qualité Publication générées dans : {output_dir}")

# ==================================================================================================
# 3. EXÉCUTION
# ==================================================================================================

def main():
    timestamp = datetime.now().strftime("%Y-%m-%d_%Hh%Mm%S")
    base_dir = "/content/drive/MyDrive/Mon_Projet" # À adapter selon ton Drive
    run_dir = os.path.join(base_dir, "runs", f"run_2D_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # --- CONFIGURATION 2D ---
    n_modes, radius = 8, 4.0
    means_1 = [[radius * math.cos(i * 2*math.pi/n_modes), radius * math.sin(i * 2*math.pi/n_modes)] for i in range(n_modes)]
    
    gmm0 = GMMParams(means=jnp.array([[0.0, 0.0]]), covs=jnp.array([[[1.0, 0.0], [0.0, 1.0]]]), weights=jnp.array([1.0]))
    gmm1 = GMMParams(means=jnp.array(means_1), covs=jnp.array([[[0.2, 0.0], [0.0, 0.2]]]*n_modes), weights=jnp.array([1.0/n_modes]*n_modes))

    cfg_base = PipelineConfig(
        gmm0=gmm0, gmm1=gmm1, T=1.0, seed=42, n_samples=20000, 
        dt_train=1e-3, dt_eval=1e-4, batch_size_train=1024, emb_dim=64, lr_init=0.001, patience=2000
    )
    
    configs = {
        "AIS": cfg_base,
        "MCD": dataclasses.replace(cfg_base, n_epochs=15000),
        "CMCD": dataclasses.replace(cfg_base, n_epochs=15000),
        "LED": dataclasses.replace(cfg_base, n_epochs=5000, patience=500)
    }

    print(f"\n🚀 Démarrage de l'expérience 2D...")
    w_ais, trj_ais = run_ais(configs["AIS"])
    (w_mcd, trj_mcd), loss_mcd = train_and_eval_mcd(configs["MCD"])
    (w_cmcd, trj_cmcd), loss_cmcd = train_and_eval_cmcd(configs["CMCD"])
    (w_led, trj_led), loss_led = train_and_eval_led(configs["LED"])

    algos_data = {
        "AIS":  {"works": w_ais,  "trajs": trj_ais,  "color": "gray",   "losses": None},
        "MCD":  {"works": w_mcd,  "trajs": trj_mcd,  "color": "blue",   "losses": loss_mcd},
        "CMCD": {"works": w_cmcd, "trajs": trj_cmcd, "color": "orange", "losses": loss_cmcd},
        "LED":  {"works": w_led,  "trajs": trj_led,  "color": "green",  "losses": loss_led}
    }

    metrics = [compute_metrics(data["works"], name) for name, data in algos_data.items()]
    print_table(metrics, run_dir)
    
    # LA VRAIE VALEUR (Calculée rigoureusement : -ln(0.2))
    true_dF = -np.log(0.2) 
    
    generate_all_plots(algos_data, true_dF, gmm1, cfg_base.n_samples, run_dir)

if __name__ == "__main__":
    main()