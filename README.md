# SafeRL-IFT-7201

Safe RL in a non-deterministic environment (Frozen Lake V1).

## Structure du projet

```
SafeRL-IFT-7201/
├── environments.py                 # Définition des environnements FrozenLake encapsulés
├── utils.py                        # Fonctions utilitaires
├── train_ppo.py                    # Script d'entraînement pour PPO
├── Visualize_ppo_results.ipynb     # Notebook pour visualiser les résultats
├── pyproject.toml                  # Configuration des dépendances (utilise `uv`)
└── README.md
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

### Entraîner un modèle PPO
```bash
# Via uv run:
uv run train_ppo.py
```

Le script affichera:
- Progression de l'entraînement (si verbose=1)
- Emplacement du modèle sauvegardé
- La politique apprise (grille avec symboles directionnels)

### Inspecter les logs tensorboard
```bash
# Visualiser les graphiques d'entraînement
tensorboard --logdir ./<env_type>_tensorboard/

# Exemple:
tensorboard --logdir ./8x8_default_tensorboard/
```
