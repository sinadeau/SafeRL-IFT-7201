"""
Entraînement d'une politique PPO sur l'environnement FrozenLake.
"""

from environments import get_env_by_type
from utils import extract_policy_grid, print_policy
from stable_baselines3 import PPO


# ==================== CONFIGURATION ====================

# Paramètres de l'expérience
SUCCESS_RATE = 9 / 10.0  # Probabilité de ne pas glisser (slippery env)
MAX_EPISODE_STEPS = 60  # Durée maximale d'un épisode (nombre de pas)
TOTAL_TIMESTEPS = 500_000  # Nombre total de pas pour l'entraînement
ENV_TYPE = "8x8_default"  # Type d'environnement: "8x8_default" ou "10x10_custom"

# Paramètres PPO
PPO_POLICY = "MlpPolicy"  # Architecture de politique: "MlpPolicy" pour réseaux de neurones
PPO_VERBOSE = 1  # Niveau de verbosité (1 = afficher les mises à jour)


# ==================== PIPELINE D'ENTRAÎNEMENT ====================

def main():
    """Entraîne une politique PPO et affiche les résultats."""
    
    # Créer l'environnement
    print(f"Création de l'environnement: {ENV_TYPE}")
    env = get_env_by_type(
        ENV_TYPE,
        max_episode_steps=MAX_EPISODE_STEPS,
        success_rate=SUCCESS_RATE
    )
    
    # Créer et entraîner le modèle
    print(f"\nEntraînement du modèle PPO...")
    print(f"  - Total timesteps: {TOTAL_TIMESTEPS:,}")
    print(f"  - Environnement: {ENV_TYPE}")
    
    model = PPO(
        PPO_POLICY,
        env,
        verbose=PPO_VERBOSE,
        tensorboard_log=f"./{ENV_TYPE}_tensorboard/"
    )
    
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        tb_log_name=f"ppo_{ENV_TYPE}"
    )
    
    # Sauvegarder le modèle
    model_path = f"ppo_{ENV_TYPE}"
    model.save(model_path)
    print(f"\nModèle sauvegardé: {model_path}")
    
    # Extraire et afficher la politique apprise
    print("\nExtraction de la politique apprise...")
    policy_grid = extract_policy_grid(env, model)
    print()
    print_policy(policy_grid)
    
    # Fermer l'environnement
    env.close()
    print("\nEntraînement terminé.")


if __name__ == "__main__":
    main()
