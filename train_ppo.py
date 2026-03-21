"""
Entraînement d'une politique PPO sur l'environnement FrozenLake.
"""

from environments import get_env_by_type
from utils import extract_policy_grid, print_policy, get_moving_averages_with_x
from utils_SB3 import MetricsCallback
from stable_baselines3 import PPO
import matplotlib.pyplot as plt ###Enlever lorsque visualisation des résultats dans Jupyter notebook
import numpy as np ###Enlever lorsque visualisation des résultats dans Jupyter notebook

# ==================== CONFIGURATION ====================

# Paramètres de l'expérience
SUCCESS_RATE = 9 / 10.0  # Probabilité de ne pas glisser (slippery env)
MAX_EPISODE_STEPS = 60  # Durée maximale d'un épisode (nombre de pas)
TOTAL_TIMESTEPS = 10_000  # Nombre total de pas pour l'entraînement
ENV_TYPE = "10x10_custom"  # Type d'environnement: "8x8_default" ou "10x10_custom"
BASE_SEED = 42

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
        success_rate=SUCCESS_RATE,
        seed = BASE_SEED ###À modifier pour rouler n répétitions de l'expérience
    )
    
    # Créer et entraîner le modèle
    print(f"\nEntraînement du modèle PPO...")
    print(f"  - Total timesteps: {TOTAL_TIMESTEPS:,}")
    print(f"  - Environnement: {ENV_TYPE}")
    
    model = PPO(
        PPO_POLICY,
        env,
        verbose=PPO_VERBOSE,
        tensorboard_log=f"./{ENV_TYPE}_tensorboard/",
        seed = BASE_SEED ###À modifier pour rouler n répétitions de l'expérience
    )
    
    callback = MetricsCallback()
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callback,
        tb_log_name=f"ppo_{BASE_SEED}" ###À modifier pour rouler n répétitions de l'expérience
    )
    
    # Sauvegarder le modèle
    model_path = f"ppo_{ENV_TYPE}_{BASE_SEED}" ###À modifier pour rouler n répétitions de l'expérience
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

    # Plot metrics (LE FAIRE DANS UN JUPYTER)    
    x_raw, x_ma, reward_mean, length_mean, fall_rate, timeout_rate, risk_mean, cum_falls = get_moving_averages_with_x(callback, window = 30)
  
    fig, axs = plt.subplots(2, 3, figsize=(12, 8))

    axs[0, 0].plot(x_ma, reward_mean)
    axs[0, 0].set_title("Reward moyen par épisode")
    axs[0, 0].set_xlabel("Épisodes")

    axs[0, 1].plot(x_ma, length_mean)
    axs[0, 1].set_title("Longueur moyenne de l'épisode")
    axs[0, 1].set_xlabel("Épisodes")

    axs[0, 2].plot(x_ma, fall_rate)
    axs[0, 2].set_title("Taux de chute")
    axs[0, 2].set_xlabel("Épisodes")

    axs[1, 0].plot(x_raw, cum_falls)
    axs[1, 0].set_title("Nombre cumulé de chutes")
    axs[1, 0].set_xlabel("Épisodes")

    axs[1, 1].plot(x_ma, risk_mean)
    axs[1, 1].set_title("Risque moyen par épisode")
    axs[1, 1].set_xlabel("Épisodes")

    axs[1, 2].plot(x_ma, timeout_rate)
    axs[1, 2].set_title("Taux de timeout")
    axs[1, 2].set_xlabel("Épisodes")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()