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

def get_moving_averages_with_x(callback, window = 30):

        rewards = np.array(callback.episode_rewards)
        lengths = np.array(callback.episode_lengths)
        falls = np.array(callback.episode_falls)
        risks = np.array(callback.episode_risks)
        successes = np.array(callback.episode_successes)
        timeouts = np.array(callback.episode_timeouts)
        cum_falls = np.array(callback.cum_falls)
        timesteps = np.array(callback.episode_end_timesteps)

        reward_mean = moving_average(rewards, window=window)
        length_mean = moving_average(lengths, window=window)
        fall_rate = moving_average(falls, window=window)
        timeout_rate = moving_average(timeouts, window=window)
        risk_mean = moving_average(risks, window=window)
        
        x_raw = np.arange(1, len(rewards) + 1)
        if len(rewards) < window:
            x_ma = x_raw
        else:
            x_ma = np.arange(window, len(rewards) + 1)

        return x_raw, x_ma, reward_mean, length_mean, fall_rate, timeout_rate, risk_mean, cum_falls


def compute_rollout_metrics(callback: 'QLearningMetricsCallback') -> dict:
    """
    Calcule les métriques rollout moyennes à partir des données d'entraînement.
    
    Args:
        callback: Instance de QLearningMetricsCallback contenant les métriques collectées
        
    Returns:
        dict: Dictionnaire contenant les métriques rollout moyennes:
            - ep_len_mean: Longueur moyenne des épisodes
            - ep_rew_mean: Récompense moyenne des épisodes
            - success_rate: Taux de succès (proportion d'épisodes avec succès)
    """
    if not callback.episode_lengths:
        return {"ep_len_mean": 0.0, "ep_rew_mean": 0.0, "success_rate": 0.0}
    
    ep_len_mean = np.mean(callback.episode_lengths)
    ep_rew_mean = np.mean(callback.episode_rewards)
    success_rate = np.mean(callback.episode_successes)
    
    return {
        "ep_len_mean": ep_len_mean,
        "ep_rew_mean": ep_rew_mean,
        "success_rate": success_rate
    }


class QLearningMetricsCallback:
    """
    Callback personnalisée pour suivre des métriques lors de l'entraînement Q-learning.
    
    Enregistre les mêmes métriques que MetricsCallback mais sans dépendre de Stable Baselines3.
    Cette classe accumule les informations pour chaque pas en mémoire et calcule les
    métriques agrégées lorsqu'un épisode se termine.
    """

    def __init__(self, verbose=0):
        self.verbose = verbose

        self.current_reward = 0.0
        self.current_length = 0
        self.current_risk = 0

        # Listes avec 1 valeur par épisode
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_falls = []
        self.episode_successes = []
        self.episode_timeouts = []
        self.episode_risks = []
        self.cum_falls = []
        self.episode_end_timesteps = []

        self.total_falls = 0
        self.num_timesteps = 0

    def on_step(self, reward: float, info: dict, done: bool, timestep: int) -> None:
        """
        Enregistre les informations d'un pas de simulation.
        
        Args:
            reward: Récompense obtenue à ce pas
            info: Dictionnaire d'information contenant is_fall, is_success, is_timeout, risk
            done: Booléen indiquant si l'épisode est terminé
            timestep: Numéro du timestep courant
        """
        self.current_reward += reward
        self.current_length += 1
        self.current_risk += info.get("risk", 0)
        self.num_timesteps = timestep

        if done:
            fall = int(info.get("is_fall", False))
            success = int(info.get("is_success", False))
            timeout = int(info.get("is_timeout", False))

            self.episode_rewards.append(self.current_reward)
            self.episode_lengths.append(self.current_length)
            self.episode_risks.append(self.current_risk)
            self.episode_falls.append(fall)
            self.episode_successes.append(success)
            self.episode_timeouts.append(timeout)

            self.total_falls += fall
            self.cum_falls.append(self.total_falls)
            self.episode_end_timesteps.append(self.num_timesteps)

            # --- Reset ---
            self.current_reward = 0.0
            self.current_length = 0
            self.current_risk = 0