# Archive — Version modulaire (gmm_*.py)

Cette archive contient la première version du pipeline, structurée en 11 modules séparés.

## Structure

| Fichier | Rôle |
|---|---|
| `gmm_config.py` | Dataclasses `GMMParams` et `PipelineConfig` |
| `gmm_physics.py` | Fonctions d'énergie NumPy (AIS) |
| `gmm_jax_physics.py` | Fonctions d'énergie JAX (MCD, CMCD, LED) |
| `gmm_networks.py` | Réseau ResNet partagé |
| `gmm_eval_utils.py` | Boucle d'évaluation en mini-batches |
| `gmm_ais.py` | AIS |
| `gmm_mcd.py` | MCD — train() + estimate() |
| `gmm_cmcd.py` | CMCD — train() + estimate() |
| `gmm_led.py` | LED — train() + estimate() |
| `gmm_metrics.py` | Métriques et tableaux comparatifs |
| `gmm_plots.py` | Visualisations |
| `run_experiment.py` | Point d'entrée principal |

## Différences avec la version actuelle (src/)

- Architecture modulaire vs fichiers centralisés
- Séparation explicite train/eval dans des fonctions distinctes
- Pas de séparation dt_train / dt_eval (version actuelle plus flexible)
- Hutchinson pour la divergence LED (version actuelle utilise jacfwd)
