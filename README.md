# SafeRL-IFT-7201

Étude comparative d'approches d'apprentissage par renforcement pour l'apprentissage de politiques sécuritaires dans des environnements non-déterministes.

## Structure du projet

```
SafeRL-IFT-7201/
├── environments.py                 # Définition des environnements FrozenLake encapsulés
├── utils.py                        # Fonctions utilitaires
├── train_ppo.py                    # Script d'entraînement pour PPO
├── train_q_learning.py             # Script d'entraînement pour Q-learning
├── generate_train_plots.py         # Script pour générer les graphiques d'entraînement
├── Visualize_results.ipynb         # Notebook pour visualiser les résultats
├── pyproject.toml                  # Configuration des dépendances (utilise `uv`)
├── README.md                       # Ce fichier
├── Figures/                        # Dossier contenant les figures générées
├── ppo_8x8_default/                # Résultats d'entraînement PPO (grille 8x8 par défaut)
├── ppo_10x10_custom/               # Résultats d'entraînement PPO (grille 10x10 personnalisée)
├── ppo_16x16_random/               # Résultats d'entraînement PPO (grille 16x16 aléatoire)
├── q_learning_8x8_default/         # Résultats d'entraînement Q-learning (grille 8x8 par défaut)
├── q_learning_10x10_custom/        # Résultats d'entraînement Q-learning (grille 10x10 personnalisée)
└── q_learning_16x16_random/        # Résultats d'entraînement Q-learning (grille 16x16 aléatoire)
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

### Entraîner un modèle PPO / Q-learning
```bash
uv run train_ppo.py
uv run train_q_learning.py
```

### Inspecter les logs tensorboard
```bash
# Visualiser les graphiques d'entraînement
tensorboard --logdir ./<algorithm>_<env_type>/

# Exemple:
tensorboard --logdir ./ppo_8x8_default/
```
