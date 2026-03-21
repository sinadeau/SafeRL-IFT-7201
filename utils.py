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
        row, col = divmod(s, ncol)
        cell = desc[row, col]
        
        # Garder les cases spéciales
        if cell in ["H", "G"]:
            policy_grid[row, col] = cell
        else:
            # Obtenir l'action préférentielle du modèle
            action, _ = model.predict(s, deterministic=True)
            policy_grid[row, col] = ARROW_MAP[int(action)]
    
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

def moving_average(x, window=20):
    """
    Calcule une moyenne glissante (moving average) sur une série de données.

    Args:
        x (array-like): Série de données (ex: rewards, longueurs d’épisodes, etc.)
        window (int): Taille de la fenêtre de lissage (nombre de points)

    Returns:
        np.ndarray:
            - Si len(x) >= window : série lissée de longueur len(x) - window + 1
            - Sinon : retourne la série originale (pas assez de points pour lisser)

    Notes:
        - Le mode "valid" garantit que seules les moyennes complètes sont conservées
          (pas de padding).
        - Chaque valeur retournée correspond à la moyenne des `window` points précédents.
    """
    if len(x) < window:
        return x
    return np.convolve(x, np.ones(window)/window, mode="valid")

#def get_moving_averages_with_x(callback, window = 30):

#    rewards = np.array(callback.episode_rewards)
#    lengths = np.array(callback.episode_lengths)
#    falls = np.array(callback.episode_falls)
#    risks = np.array(callback.episode_risks)    
#    timeouts = np.array(callback.episode_timeouts)
#    cum_falls = np.array(callback.cum_falls)
#    #successes = np.array(callback.episode_successes)
#    #timesteps = np.array(callback.episode_end_timesteps)

#    reward_mean = moving_average(rewards, window=window)
#    length_mean = moving_average(lengths, window=window)
#    fall_rate = moving_average(falls, window=window)
#    timeout_rate = moving_average(timeouts, window=window)
#    risk_mean = moving_average(risks, window=window)
        
#    x_raw = np.arange(1, len(rewards) + 1)
#    if len(rewards) < window:
#        x_ma = x_raw
#    else:
#        x_ma = np.arange(window, len(rewards) + 1)

#    return x_raw, x_ma, reward_mean, length_mean, fall_rate, timeout_rate, risk_mean, cum_falls

def get_moving_averages_with_x(data, window=30):
    rewards = np.array(data["episode_rewards"])
    lengths = np.array(data["episode_lengths"])
    falls = np.array(data["episode_falls"])
    risks = np.array(data["episode_risks"])
    timeouts = np.array(data["episode_timeouts"])
    cum_falls = np.array(data["cum_falls"])
    #successes = np.array(data["episode_successes"])
    #timesteps = np.array(data["episode_end_timesteps"])

    reward_mean = moving_average(rewards, window)
    length_mean = moving_average(lengths, window)
    fall_rate = moving_average(falls, window)
    timeout_rate = moving_average(timeouts, window)
    risk_mean = moving_average(risks, window)

    x_raw = np.arange(1, len(rewards) + 1)

    if len(rewards) < window:
        x_ma = x_raw
    else:
        x_ma = np.arange(window, len(rewards) + 1)

    return x_raw, x_ma, reward_mean, length_mean, fall_rate, timeout_rate, risk_mean, cum_falls


def save_callback_metrics_npz(callback, filepath):
    """
    Sauvegarde les métriques du callback dans un fichier .npz
    """

    np.savez(
        filepath,
        episode_rewards=np.array(callback.episode_rewards),
        episode_lengths=np.array(callback.episode_lengths),
        episode_falls=np.array(callback.episode_falls),
        episode_risks=np.array(callback.episode_risks),        
        episode_timeouts=np.array(callback.episode_timeouts),
        cum_falls=np.array(callback.cum_falls),
        episode_successes=np.array(callback.episode_successes),
        episode_end_timesteps=np.array(callback.episode_end_timesteps),
    )

def load_callback_metrics_npz(filepath):
    """
    Charge les métriques depuis un fichier .npz
    """
    data = np.load(filepath)
    return {key: data[key] for key in data.files}