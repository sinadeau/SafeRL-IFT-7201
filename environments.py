"""
Définition des environnements FrozenLake.
"""

import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.monitor import Monitor


def create_env_8x8_default(max_episode_steps: int = 60, success_rate: float = 0.9) -> gym.Env:
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
    env = Monitor(env)
    
    return env


def create_env_10x10_custom(max_episode_steps: int = 60, success_rate: float = 0.9) -> gym.Env:
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
    env = Monitor(env)
    
    return env


def get_env_by_type(
    env_type: str,
    max_episode_steps: int = 60,
    success_rate: float = 0.9
) -> gym.Env:
    """
    Récupère un environnement selon son type.
    
    Args:
        env_type: Type d'environnement ("8x8_default" ou "10x10_custom")
        max_episode_steps: Durée maximale d'un épisode
        success_rate: Probabilité de ne pas glisser
        
    Returns:
        gym.Env: L'environnement demandé
        
    Raises:
        ValueError: Si le type d'environnement n'est pas reconnu
    """
    if env_type == "8x8_default":
        return create_env_8x8_default(max_episode_steps, success_rate)
    elif env_type == "10x10_custom":
        return create_env_10x10_custom(max_episode_steps, success_rate)
    else:
        raise ValueError(f"Type d'environnement inconnu: {env_type}")
