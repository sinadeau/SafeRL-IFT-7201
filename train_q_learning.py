"""
Entraînement d'une politique Q-learning pour l'environnement FrozenLake.
"""

import os
import numpy as np
import torch
from tqdm import tqdm
from environments import create_env
from utils import (QLearningMetricsCallback, save_callback_metrics_npz,
                   load_callback_metrics_npz, run_training_repetitions,
                   average_smoothed_metrics_across_runs, log_averaged_metrics_to_tensorboard,
                   log_run_metrics_to_tensorboard, evaluate_policy_with_metrics,
                   average_eval_metrics_across_runs, eval_metrics_to_dataframe)

class Qlearning:
    """
    Classe Q-learning pour l'entraînement d'une politique sur un environnement Gymnasium. Le code pour
    cette classe a été adapté à partir de l'exemple d'entraînement Q-learning sur FrozenLake disponible
    sur le site de Gymnasium: https://gymnasium.farama.org/tutorials/training_agents/frozenlake_q_learning/
    """

    def __init__(self, learning_rate, gamma, state_size, action_size, epsilon=0.1, epsilon_min=0.01, verbose=0, seed=42):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate_init = learning_rate
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon_init = epsilon
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.verbose = verbose
        self.rng = np.random.default_rng(seed=seed)

        self.reset_qtable()

    def update(self, state, action, reward, new_state):
        """Mise à jour Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]"""
        delta = (
            reward
            + self.gamma * np.max(self.qtable[new_state, :])
            - self.qtable[state, action]
        )
        q_update = self.qtable[state, action] + self.learning_rate * delta
        return q_update

    def reset_qtable(self):
        """Réinitialise la Q-table."""
        self.qtable = np.zeros((self.state_size, self.action_size))

    def choose_action(self, state, action_space):
        """Choisit une action selon epsilon-greedy."""
        if self.rng.uniform(0, 1) < self.epsilon:
            return action_space.sample()
        else:
            max_ids = np.where(self.qtable[state, :] == np.max(self.qtable[state, :]))[0]
            return self.rng.choice(max_ids)

    def learn(self, env, total_episodes, callback = None):
        """
        Boucle d'entraînement Q-learning contrôlée par le nombre d'épisodes.
        
        Args:
            env: Environnement Gymnasium
            total_episodes: Nombre total d'épisodes pour l'entraînement
            callback: Callback à appeler à chaque pas de temps
        """
        timestep = 0
        episode = 0
        pbar = tqdm(total=total_episodes, desc="Training Q-learning", unit="episodes")
        decay_factor = 0.9995  # Facteur de décroissance pour learning rate et epsilon
        
        while episode < total_episodes:
            state = env.reset()[0]
            done = False
            
            while not done:
                # Choisir une action
                action = self.choose_action(state, env.action_space)
                
                # Exécuter l'action
                new_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                # Mettre à jour la Q-table
                self.qtable[state, action] = self.update(
                    state, action, reward, new_state
                )
                
                # Enregistrer les métriques
                callback.on_step(reward, info, done, timestep)
                
                # Décay du learning rate et epsilon
                self.learning_rate = self.learning_rate_init * (decay_factor ** episode)
                self.epsilon = max(self.epsilon_min, self.epsilon_init * (decay_factor ** episode))
                
                # Incrémenter le timestep
                timestep += 1
                
                # Mettre à jour l'état
                state = new_state
            episode += 1
            pbar.update(1)
        pbar.close()

    def save(self, path):
        """Sauvegarde la Q-table dans un fichier."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, self.qtable)
    
    def predict(self, state, deterministic=True):
        """
        Prédit l'action optimale pour un état donné basée sur la Q-table.
        
        Cette méthode rend Qlearning compatible avec extract_policy_grid.
        
        Args:
            state: État scalaire (entier) ou array numpy
            deterministic: Inutilisé, présent pour compatibilité avec SB3
            
        Returns:
            tuple: (action, None) - action est l'argmax de la Q-table
        """
        if isinstance(state, np.ndarray):
            # Accepte les observations scalaires ou tableaux de taille 1
            state = int(np.asarray(state).reshape(-1)[0])
        else:
            state = int(state)
        action = np.argmax(self.qtable[state, :])
        return action, None

# ==================== CONFIGURATION ====================

# Paramètres de l'expérience
SUCCESS_RATE = 9 / 10.0        # Probabilité de ne pas glisser (slippery env)
MAX_EPISODE_STEPS = 60         # Durée maximale d'un épisode (nombre de pas)
TOTAL_EPISODES = 150_000       # Nombre total d'épisodes par répétition
ENV_TYPE = "16x16_random"      # Type d'environnement: "8x8_default", "10x10_custom" ou "16x16_random"
BASE_SEED = 40                 # Graine de base ; la répétition i utilise BASE_SEED + i
N_REPETITIONS = 10             # Nombre de répétitions d'entraînement
WINDOW = 100                   # Fenêtre de lissage pour les métriques moyennées

# Paramètres d'évaluation
N_EVAL_EPISODES = 100  # Nombre d'épisodes par répétition pour l'évaluation finale de la politique apprise

# Paramètres Q-learning
LEARNING_RATE = 1     # Taux d'apprentissage initial (alpha_0)
GAMMA = 0.99          # Taux d'actualisation (discount factor)
EPSILON = 1           # Probabilité d'exploration initiale
EPSILON_MIN = 0.001   # Probabilité d'exploration minimale
QLEARNING_VERBOSE = 1 # Niveau de verbosité (1 = afficher les mises à jour)

DEVICE = torch.device('cpu')

# ==================== ENTRAÎNEMENT D'UNE RÉPÉTITION ====================

def train_one_q_learning(seed):
    """Entraîne une répétition Q-learning avec la graine donnée et sauvegarde les artéfacts."""
    output_dir = f"q_learning_{ENV_TYPE}"
    os.makedirs(output_dir, exist_ok=True)

    env = create_env(ENV_TYPE, max_episode_steps=MAX_EPISODE_STEPS,
                     success_rate=SUCCESS_RATE, seed=seed)
    state_size = env.observation_space.n
    action_size = env.action_space.n

    learner = Qlearning(
        learning_rate=LEARNING_RATE, gamma=GAMMA,
        state_size=state_size, action_size=action_size,
        epsilon=EPSILON, epsilon_min=EPSILON_MIN, verbose=QLEARNING_VERBOSE, seed=seed
    )
    callback = QLearningMetricsCallback(verbose=QLEARNING_VERBOSE)

    learner.learn(env=env, total_episodes=TOTAL_EPISODES, callback=callback)
    learner.save(f"{output_dir}/model_{seed}.npy")
    callback_path = f"{output_dir}/callback_{seed}.npz"
    save_callback_metrics_npz(callback, callback_path)
    callback_metrics = load_callback_metrics_npz(callback_path)

    # Évaluation de la politique apprise
    eval_metrics = evaluate_policy_with_metrics(learner, env, n_eval_episodes=N_EVAL_EPISODES, deterministic=True)

    env.close()
    print(f"\nRépétition terminée (seed={seed}).")
    return callback_metrics, eval_metrics


# ==================== PIPELINE D'ENTRAÎNEMENT ====================

def main():
    """Lance N_REPETITIONS entraînements Q-learning en parallèle et calcule les métriques moyennées."""
    print(f"Entraînement Q-learning — {N_REPETITIONS} répétition(s), environnement: {ENV_TYPE}")

    output_dir = f"q_learning_{ENV_TYPE}"

    # Lancer toutes les répétitions en parallèle (ThreadPoolExecutor)
    results = run_training_repetitions(train_one_q_learning, N_REPETITIONS, BASE_SEED, DEVICE)
    callbacks_data, eval_data = zip(*results)
    callbacks_data = list(callbacks_data)
    eval_data = list(eval_data)

    # Écrire les métriques par épisode dans TensorBoard pour chaque répétition (séquentiel, thread principal)
    print("Écriture des métriques par épisode dans TensorBoard...")
    for i, data in enumerate(callbacks_data):
        seed = BASE_SEED + i
        n_episodes = len(data["episode_rewards"])
        if n_episodes != TOTAL_EPISODES:
            raise ValueError(
                f"Répétition seed={seed}: {n_episodes} épisodes au lieu de {TOTAL_EPISODES}."
            )
        log_run_metrics_to_tensorboard(data, f"{output_dir}/tensorboard/q_learning_{seed}")
        print(f"  TensorBoard écrit pour seed={seed} ({n_episodes} épisodes)")

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
    avg_path = f"{output_dir}/averaged_metrics.npz"
    np.savez(avg_path, **averaged)
    # Métriques d'évaluation finale (sauvegarder sous format pandas DataFrame)
    eval_path = f"{output_dir}/averaged_eval_metrics.csv"
    df_eval = eval_metrics_to_dataframe(averaged_eval, std_eval)
    df_eval.to_csv(eval_path, index=False)

    print(f"Métriques moyennées sauvegardées: {avg_path}")

    # Enregistrer les métriques d'entraînement moyennées dans TensorBoard
    tb_dir = f"{output_dir}/tensorboard/q_learning_averaged"
    log_averaged_metrics_to_tensorboard(averaged, tb_dir)
    print(f"Métriques moyennées enregistrées dans TensorBoard: {tb_dir}")

    print("\nEntraînement et évaluation terminé.")


if __name__ == "__main__":
    main()
