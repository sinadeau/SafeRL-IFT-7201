"""
Fonctions utilitaires.
"""

import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from stable_baselines3.common.policies import BasePolicy


# Correspondance actions -> symboles de direction
ARROW_MAP = {
    0: "←",  # Gauche
    1: "↓",  # Bas
    2: "→",  # Droite
    3: "↑"   # Haut
}


def extract_policy_grid(env: gym.Env, model: BasePolicy) -> np.ndarray:
    """
    Extrait la politique apprise par le modèle sous forme de grille.
    
    Pour chaque état de l'environnement FrozenLake, récupère l'action préférentielle
    du modèle et la représente par un symbole de direction. Les cases spéciales
    (trous, objectif) restent identifiées.
    
    Args:
        env: Environnement FrozenLake encapsulé
        model: Modèle entraîné (ex. PPO, DQN) ayant une méthode predict()
        
    Returns:
        np.ndarray: Grille 2D contenant les symboles de politique ou d'état spécial
    """
    # Récupérer la description de la carte
    desc = env.unwrapped.desc.astype(str)
    nrow, ncol = desc.shape
    policy_grid = np.empty((nrow, ncol), dtype=object)
    
    # Parcourir tous les états
    for s in range(nrow * ncol):
        r, c = divmod(s, ncol)
        cell = desc[r, c]
        
        # Garder les cases spéciales
        if cell in ["H", "G"]:
            policy_grid[r, c] = cell
        else:
            # Obtenir l'action préférentielle du modèle
            action, _ = model.predict(s, deterministic=True)
            policy_grid[r, c] = ARROW_MAP[int(action)]
    
    return policy_grid


def print_policy(policy_grid: np.ndarray) -> None:
    """
    Affiche la politique apprise dans la console.
    
    Args:
        policy_grid: Grille de politique obtenue via extract_policy_grid()
    """
    print("Politique apprise:")
    for row in policy_grid:
        print("  ".join(row))
