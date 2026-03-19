"""
Définition des environnements FrozenLake.
"""

import numpy as np
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.monitor import Monitor


def create_env_8x8_default(max_episode_steps: int = 60, success_rate: float = 0.9, seed: int = 42) -> gym.Env:
    """
    Crée un environnement FrozenLake 8x8 avec configuration par défaut.
    
    Configuration inspirée de Zhu et al. (2025):
    - Grille 8x8
    - Trous (H) positionnés stratégiquement
    - Objectif (G) défini
    - Récompenses: Goal=+1, Hole=-0.1, Frozen=-0.01
    
    Args:
        max_episode_steps: Durée maximale d'un épisode (nombre de pas)
        success_rate: Probabilité de ne pas glisser (1.0 = déterministe)
        seed: graine pour la génération aléatoire
        
    Returns:
        gym.Env: Environnement encapsulé avec TimeLimit et Monitor
    """
    reward_schedule = (1, -0.1, -0.01)  # (Goal, Hole, Frozen)
    
    desc = [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG",
    ]
    
    env = gym.make(
        "FrozenLake-v1",
        desc=desc,
        is_slippery=True,
        success_rate=success_rate,
        reward_schedule=reward_schedule
    )
    
    env = TimeLimit(env, max_episode_steps=max_episode_steps)
    env = FrozenLakeInfoWrapper(env)
    env = Monitor(env) #Pour log métriques standard dans tensorboard
    env.reset(seed=seed)
    
    
    return env


def create_env_10x10_custom(max_episode_steps: int = 60, success_rate: float = 0.9, seed: int = 42) -> gym.Env:
    """
    Crée un environnement FrozenLake 10x10 personnalisé.
    
    Configuration adaptée pour tester les performances sur des environnements 
    plus grands avec une distribution différente de trous.
    
    - Grille 10x10
    - Configuration déduite empiriquement
    - Récompenses: Goal=+1, Hole=-0.5, Frozen=-0.01
    
    Args:
        max_episode_steps: Durée maximale d'un épisode (nombre de pas)
        success_rate: Probabilité de ne pas glisser (1.0 = déterministe)
        seed: graine pour la génération aléatoire
        
    Returns:
        gym.Env: Environnement encapsulé avec TimeLimit et Monitor
    """
    reward_schedule = (1, -0.5, -0.01)  # (Goal, Hole, Frozen)
    
    desc = [
        "FFFFFFFFFF",
        "FFFFFFFFFF",
        "HFHFFFFFFF",
        "FFFFHFFFFF",
        "HFFFFFFHFF",
        "HFFFHFFFFF",
        "HFFFFFFHFF",
        "HFHFHFFFFF",
        "FFFFFFFFFF",
        "GFFFFFFFFF",
    ]
    
    env = gym.make(
        "FrozenLake-v1",
        desc=desc,
        is_slippery=True,
        success_rate=success_rate,
        reward_schedule=reward_schedule
    )
    
    env = TimeLimit(env, max_episode_steps=max_episode_steps)
    env = FrozenLakeInfoWrapper(env)
    env = Monitor(env) #Pour log métriques standard dans tensorboard
    env.reset(seed=seed)
    
    return env


def get_env_by_type(
    env_type: str,
    max_episode_steps: int = 60,
    success_rate: float = 0.9,
    seed: int = 42
) -> gym.Env:
    """
    Récupère un environnement selon son type.
    
    Args:
        env_type: Type d'environnement ("8x8_default" ou "10x10_custom")
        max_episode_steps: Durée maximale d'un épisode
        success_rate: Probabilité de ne pas glisser
        seed: graine pour la génération aléatoire
        
    Returns:
        gym.Env: L'environnement demandé
        
    Raises:
        ValueError: Si le type d'environnement n'est pas reconnu
    """
    if env_type == "8x8_default":
        return create_env_8x8_default(max_episode_steps, success_rate, seed)
    elif env_type == "10x10_custom":
        return create_env_10x10_custom(max_episode_steps, success_rate, seed)
    else:
        raise ValueError(f"Type d'environnement inconnu: {env_type}")


class FrozenLakeInfoWrapper(gym.Wrapper):
    """
        Wrapper pour enrichir l'environnement FrozenLake avec des informations
        supplémentaires utiles pour l'analyse des performances et de la sécurité.

        Ce wrapper ajoute dans le dictionnaire `info` des indicateurs permettant de :
        - distinguer les types de terminaison d'épisode (succès, chute, timeout)
        - associer à chaque état un niveau de risque basé sur la proximité des trous

        Le risque est défini comme une fonction du nombre de trous adjacents à la case courante. 
        Chaque trou adjacent contribue de manière additive au risque :
            risque = 2 × (nombre de trous adjacents)

        Informations ajoutées à `info` à chaque pas :
            - is_fall (bool) : True si l'agent est tombé dans un trou
            - is_success (bool) : True si l'agent a atteint l'objectif
            - is_timeout (bool) : True si l'épisode est terminé par dépassement du nombre de pas
            - risk (int) : niveau de risque de la case courante

        Args:
            env (gym.Env): Environnement FrozenLake à encapsuler

        Returns:
            gym.Env: Environnement enrichi avec des informations supplémentaires dans `info`
    """

    def __init__(self, env):
        super().__init__(env)

        base_env = env.unwrapped
        self.desc = base_env.desc
        self.desc_str = self.desc.astype(str)
        self.nrow, self.ncol = self.desc.shape

        # Pré-calcul du risque pour chaque case        
        self.risk_map = np.zeros((self.nrow, self.ncol), dtype=np.int32)

        for row in range(self.nrow):
            for col in range(self.ncol):
                holes = 0

                #Tile au dessus
                if row > 0 and self.desc_str[row - 1, col] == "H":
                    holes += 1
                #Tile en dessous
                if row < self.nrow - 1 and self.desc_str[row + 1, col] == "H":
                    holes += 1
                #Tile à gauche
                if col > 0 and self.desc_str[row, col - 1] == "H":
                    holes += 1
                #Tile à droite
                if col < self.ncol - 1 and self.desc_str[row, col + 1] == "H":
                    holes += 1

                self.risk_map[row, col] = 2 * holes

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        info["is_fall"] = False
        info["is_success"] = False
        info["is_timeout"] = bool(truncated)

        row, col = divmod(obs, self.ncol)
        tile = self.desc_str[row, col]

        info["risk"] = int(self.risk_map[row, col])

        if terminated:
            if tile == "H":
                info["is_fall"] = True
            elif tile == "G":
                info["is_success"] = True

        return obs, reward, terminated, truncated, info
    
    