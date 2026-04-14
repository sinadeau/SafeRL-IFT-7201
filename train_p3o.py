"""
Entraînement d'une politique P3O sur l'environnement FrozenLake.
"""

import os
import numpy as np
from environments import create_env
from utils import (save_callback_metrics_npz, load_callback_metrics_npz,
                   run_training_repetitions, average_smoothed_metrics_across_runs,
                   log_averaged_metrics_to_tensorboard, 
                   evaluate_policy_with_metrics, average_eval_metrics_across_runs, eval_metrics_to_dataframe)
from utils import SB3MetricsCallback
from stable_baselines3.common.callbacks import StopTrainingOnMaxEpisodes
from model_p3o import P3O
import torch

DEVICE = torch.device('cpu')

# ==================== CONFIGURATION ====================

# Paramètres de l'expérience (entraînement et environnement)
SUCCESS_RATE = 9 / 10.0        # Probabilité de ne pas glisser (slippery env)
MAX_EPISODE_STEPS = 100         # Durée maximale d'un épisode (nombre de pas)
TOTAL_EPISODES = 50_000       # Nombre total d'épisodes par répétition
ENV_TYPE = "10x10_custom"      # Type d'environnement: "8x8_default", "10x10_custom" ou "16x16_random"
BASE_SEED = 40                 # Graine de base ; la répétition i utilise BASE_SEED + i
N_REPETITIONS = 10             # Nombre de répétitions d'entraînement
WINDOW = 100                   # Fenêtre de lissage pour les métriques moyennées

# Paramètres d'évaluation
N_EVAL_EPISODES = 100  # Nombre d'épisodes par répétition pour l'évaluation finale de la politique apprise

# Paramètres P3O
P3O_REWARD_TYPE = "CMDP"  # Formulation des rewards pour CMDP. Pour un CMDP la récompense de tomber dans un trou est de 0 pour dissocier le coût de la fonction de récompense.
P3O_KAPPA = 20            # Kappa utilisé dans la lost de coût pour la politique
P3O_COST_LIMIT = 1             # Limite de coût cumulatif par épisode autorisée (le d dans: Jc <= d)
P3O_POLICY = "MlpPolicy"  # Architecture de politique: "MlpPolicy" pour réseaux de neurones
P3O_VERBOSE = 0           # Niveau de verbosité (1 = afficher les mises à jour)

# ==================== ENTRAÎNEMENT D'UNE RÉPÉTITION ====================

def train_one_p3o(seed):
    """Entraîne une répétition P3O avec la graine donnée et sauvegarde les artéfacts."""
    output_dir = f"p3o_{ENV_TYPE}"
    os.makedirs(output_dir, exist_ok=True)

    env = create_env(ENV_TYPE, 
                     reward_type = P3O_REWARD_TYPE,
                     max_episode_steps=MAX_EPISODE_STEPS,
                     success_rate=SUCCESS_RATE, seed=seed)
    
    # Définition architecturale : 2 couches cachées de 16 neurones chacune, activation ReLU
    policy_kwargs = dict(activation_fn=torch.nn.ReLU,
                     net_arch=dict(pi=[16, 16], vf=[16, 16]))
    
    model = P3O(P3O_POLICY, env, verbose=P3O_VERBOSE,
                tensorboard_log=f"./{output_dir}/tensorboard/", seed=seed, policy_kwargs=policy_kwargs,
                kappa = P3O_KAPPA,
                cost_limit = P3O_COST_LIMIT,
                device=DEVICE)

    # Entraînement avec callback pour collecter les métriques et arrêter après un nombre d'épisodes
    callback = SB3MetricsCallback()
    model.learn(
        total_timesteps=int(1e10),  # Arrêté par StopTrainingOnMaxEpisodes
        callback=[callback, StopTrainingOnMaxEpisodes(max_episodes=TOTAL_EPISODES)],
        tb_log_name=f"p3o_{seed}"
    )

    model.save(f"{output_dir}/model_{seed}")
    callback_path = f"{output_dir}/callback_{seed}.npz"
    save_callback_metrics_npz(callback, callback_path)
    callback_metrics = load_callback_metrics_npz(callback_path)

    # Évaluation de la politique apprise
    eval_metrics = evaluate_policy_with_metrics(model, env, n_eval_episodes=N_EVAL_EPISODES, deterministic=True)

    env.close()
    print(f"\nRépétition terminée (seed={seed}).")
    return callback_metrics, eval_metrics


# ==================== PIPELINE D'ENTRAÎNEMENT ====================

def main():
    """Lance N_REPETITIONS entraînements P3O en parallèle et calcule les métriques moyennées."""
    print(f"Entraînement P3O — {N_REPETITIONS} répétition(s), environnement: {ENV_TYPE}, device: {DEVICE}")

    # Lancer toutes les répétitions en parallèle (ThreadPoolExecutor)
    results = run_training_repetitions(train_one_p3o, N_REPETITIONS, BASE_SEED, DEVICE)
    callbacks_data, eval_data = zip(*results)
    callbacks_data = list(callbacks_data)
    eval_data = list(eval_data)

    # Lisser et moyenner les métriques d'entraînement sur toutes les répétitions
    averaged = average_smoothed_metrics_across_runs(callbacks_data, window=WINDOW)

    # Moyenner les métriques d'évaluation finale sur toutes les répétitions
    averaged_eval, std_eval = average_eval_metrics_across_runs(eval_data)
    print("\nMétriques d'évaluation finale moyennées sur les répétitions:")
    for k, v in averaged_eval.items():
        std_k = std_eval.get(f"{k}_std", 0)
        print(f"  {k}: {v:.4f} ± {std_k:.4f}")        

    # Sauvegarder les métriques moyennées
    # Métriques d'entraînement
    output_dir = f"p3o_{ENV_TYPE}"
    avg_path = f"{output_dir}/averaged_metrics.npz"
    np.savez(avg_path, **averaged)
    # Métriques d'évaluation finale (sauvegarder sous format pandas DataFrame)
    eval_path = f"{output_dir}/averaged_eval_metrics.csv"
    df_eval = eval_metrics_to_dataframe(averaged_eval, std_eval)
    df_eval.to_csv(eval_path, index=False)

    print(f"Métriques moyennées sauvegardées: {avg_path}")

    # Enregistrer les métriques d'entraînement moyennées dans TensorBoard
    tb_dir = f"{output_dir}/tensorboard/p3o_averaged"
    log_averaged_metrics_to_tensorboard(averaged, tb_dir)
    print(f"Métriques moyennées enregistrées dans TensorBoard: {tb_dir}")

    print("\nEntraînement et évaluation terminé.")


if __name__ == "__main__":
    main()