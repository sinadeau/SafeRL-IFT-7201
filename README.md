# SafeRL-IFT-7201

Étude comparative d'approches d'apprentissage par renforcement pour l'apprentissage de politiques sécuritaires dans des environnements non-déterministes.

## Structure du projet

```
SafeRL-IFT-7201/
├── environments.py                             # Définition des environnements FrozenLake encapsulés
├── utils.py                                    # Fonctions utilitaires
├── train_ppo.py                                # Script d'entraînement pour PPO
├── train_q_learning.py                         # Script d'entraînement pour Q-learning
├── train_p3o.py                                # Script d'entraînement pour P3O
├── model_p3o.py                                # Implémentation de l'algorithme P3O (Penalized Proximal Policy Optimization) basé sur l'architecture de stable-baselines3.
├── generate_train_plots.py                     # Script pour générer les graphiques d'entraînement
├── generate_train_plots_param_search.py        # Script pour générer les graphiques de recherche d'hyperparamètres
├── Visualize_environments_and_policies.ipynb   # Notebook pour visualiser les cas d'études et politiques apprises
├── pyproject.toml                              # Configuration des dépendances (utilise `uv`)
├── README.md                                   # Ce fichier
```

## Installation des dépendances

Ce projet utilise **`uv`** pour la gestion des dépendances. Assurez-vous d'avoir `uv` installé avant de suivre les étapes ci-dessous.

### Prérequis
- Python ≥ 3.12
- `uv` installé ([documentation](https://docs.astral.sh/uv/))

### Installation
```bash
# Installer les dépendances dans l'environnement virtuel du projet
uv sync
```

## Exécution

### Entraîner un modèle PPO / Q-learning / P3O
```bash
uv run train_ppo.py
uv run train_q_learning.py
uv run train_p3o.py
```

### Inspecter les logs tensorboard
```bash
# Visualiser les graphiques d'entraînement
tensorboard --logdir ./<algorithm>_<env_type>/

# Exemple:
tensorboard --logdir ./ppo_8x8_default/
```
