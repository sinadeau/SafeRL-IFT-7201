"""
Entraînement d'une politique Q-learning pour l'environnement FrozenLake.
"""

import os
import numpy as np
from tqdm import tqdm
from environments import create_env
from utils import extract_policy_grid, print_policy, get_moving_averages_with_x, compute_rollout_metrics, QLearningMetricsCallback, save_callback_metrics_npz
from stable_baselines3.common.logger import configure
import matplotlib.pyplot as plt

class Qlearning:
    """
    Classe Q-learning pour l'entraînement d'une politique sur un environnement Gymnasium. Le code pour
    cette classe a été adapté à partir de l'exemple d'entraînement Q-learning sur FrozenLake disponible
    sur le site de Gymnasium: https://gymnasium.farama.org/tutorials/training_agents/frozenlake_q_learning/
    """

    def __init__(self, learning_rate, gamma, state_size, action_size, epsilon=0.1, verbose=0, seed=42):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate_init = learning_rate
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon_init = epsilon
        self.epsilon = epsilon
        self.epsilon_min = 0.01
        self.verbose = verbose
        self.logger = None
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
        decay_factor = 0.992  # Facteur de décroissance pour learning rate et epsilon
        
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
                
                # Logger les métriques custom et rollout
                if done:
                    rollout_metrics = compute_rollout_metrics(callback)
                    if self.logger:
                        self.logger.record("custom/episode_reward", callback.episode_rewards[-1])
                        self.logger.record("custom/episode_length", callback.episode_lengths[-1])
                        self.logger.record("custom/episode_risk", callback.episode_risks[-1])
                        self.logger.record("custom/fall", callback.episode_falls[-1])
                        self.logger.record("custom/success", callback.episode_successes[-1])
                        self.logger.record("custom/timeout", callback.episode_timeouts[-1])
                        self.logger.record("custom/cumulative_falls", callback.cum_falls[-1])
                        self.logger.record("rollout/ep_len_mean", rollout_metrics["ep_len_mean"])
                        self.logger.record("rollout/ep_rew_mean", rollout_metrics["ep_rew_mean"])
                        self.logger.record("rollout/success_rate", rollout_metrics["success_rate"])
                        self.logger.record("learning/learning_rate", self.learning_rate)
                        self.logger.record("learning/epsilon", self.epsilon)
                        self.logger.dump(step=timestep)
                
                # Incrémenter le timestep
                timestep += 1
                
                # Mettre à jour l'état
                state = new_state
            episode += 1
            pbar.update(1)
        pbar.close()
        
        # Fermer le logger
        if self.logger:
            self.logger.close()

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
        if isinstance(state, np.ndarray) and state.shape == ():
            state = int(state)
        action = np.argmax(self.qtable[state, :])
        return action, None

# ==================== CONFIGURATION ====================

# Paramètres de l'expérience
SUCCESS_RATE = 9 / 10.0  # Probabilité de ne pas glisser (slippery env)
MAX_EPISODE_STEPS = 60  # Durée maximale d'un épisode (nombre de pas)
TOTAL_EPISODES = 7_000  # Nombre total d'épisodes pour l'entraînement
ENV_TYPE = "10x10_custom"  # Type d'environnement: "8x8_default" ou "10x10_custom"
BASE_SEED = 42

# Paramètres Q-learning
LEARNING_RATE = 1  # Taux d'apprentissage initial (alpha_0)
GAMMA = 0.99  # Taux d'actualisation (discount factor)
EPSILON = 1  # Probabilité d'exploration initiale
QLEARNING_VERBOSE = 1  # Niveau de verbosité (1 = afficher les mises à jour)


# ==================== PIPELINE D'ENTRAÎNEMENT ====================

def main():
    """Entraîne une politique Q-learning et affiche les résultats."""
    
    # Créer l'environnement
    print(f"Création de l'environnement: {ENV_TYPE}")
    env = create_env(
        ENV_TYPE,
        max_episode_steps=MAX_EPISODE_STEPS,
        success_rate=SUCCESS_RATE,
        seed=BASE_SEED ###À modifier pour rouler n répétitions de l'expérience
    )
    
    # Récupérer la taille de l'espace d'états et d'actions
    state_size = env.observation_space.n
    action_size = env.action_space.n
    
    # Créer le modèle Q-learning
    print(f"\nInitialisation du modèle Q-learning...")
    print(f"  - Total episodes: {TOTAL_EPISODES:,}")
    print(f"  - Environnement: {ENV_TYPE}")
    
    learner = Qlearning(
        learning_rate=LEARNING_RATE,
        gamma=GAMMA,
        state_size=state_size,
        action_size=action_size,
        epsilon=EPSILON,
        verbose=QLEARNING_VERBOSE,
        seed=BASE_SEED ###À modifier pour rouler n répétitions de l'expérience
    )
    
    # Créer la callback pour enregistrer les métriques
    callback = QLearningMetricsCallback(verbose=QLEARNING_VERBOSE)
    
    # Configurer le logger TensorBoard
    learner.logger = configure(
        folder=f"./{ENV_TYPE}_tensorboard/q_learning_{BASE_SEED}",
        format_strings=["tensorboard"]
    )
    
    # Entraîner le modèle
    print(f"\nEntraînement du modèle Q-learning...")
    learner.learn(
        env=env,
        total_episodes=TOTAL_EPISODES,
        callback=callback
    )
    
    # Sauvegarder le modèle
    model_path = f"q_learning_{ENV_TYPE}/model_{BASE_SEED}.npy" ###À modifier pour rouler n répétitions de l'expérience
    learner.save(model_path)
    print(f"\nModèle sauvegardé: {model_path}")

    # Sauvegarder le callback
    callback_path = f"q_learning_{ENV_TYPE}/callback_{BASE_SEED}.npz" ###À modifier pour rouler n répétitions de l'expérience
    save_callback_metrics_npz(callback, callback_path)
    print(f"\nCallback sauvegardé: {callback_path}")
    
    # Extraire et afficher la politique apprise
    print("\nExtraction de la politique apprise...")
    policy_grid = extract_policy_grid(env, learner)
    print()
    print_policy(policy_grid)
    
    # Fermer l'environnement
    env.close()
    print("\nEntraînement terminé.")

if __name__ == "__main__":
    main()
