"""
Entraînement d'une politique Q-learning pour l'environnement FrozenLake.
"""

import numpy as np
import gymnasium as gym
from tqdm import tqdm
from environments import get_env_by_type
from utils import extract_policy_grid, print_policy, get_moving_averages_with_x, compute_rollout_metrics, QLearningMetricsCallback
import matplotlib.pyplot as plt

rng = np.random.default_rng()

class Qlearning:
    """
    Classe Q-learning pour l'entraînement d'une politique sur un environnement Gymnasium. Le code pour
    cette classe a été adapté à partir de l'exemple d'entraînement Q-learning sur FrozenLake disponible
    sur le site de Gymnasium: https://gymnasium.farama.org/tutorials/training_agents/frozenlake_q_learning/
    """

    def __init__(self, learning_rate, gamma, state_size, action_size, verbose=0):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.verbose = verbose

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

    def learn(self, env, total_timesteps, explorer, callback):
        """
        Boucle d'entraînement Q-learning contrôlée par le nombre de timesteps.
        
        Args:
            env: Environnement Gymnasium
            total_timesteps: Nombre total de pas d'entraînement
            explorer: Instance d'EpsilonGreedy pour la stratégie d'exploration
            callback: Instance de QLearningMetricsCallback pour enregistrer les métriques
        """
        timestep = 0
        
        # Barre de progression
        pbar = tqdm(total=total_timesteps, desc="Training Q-learning", unit="timestep")
        
        while timestep < total_timesteps:
            state = env.reset()[0]  # Réinitialiser l'environnement
            done = False
            
            while not done and timestep < total_timesteps:
                # Choisir une action selon la politique epsilon-greedy
                action = explorer.choose_action(
                    action_space=env.action_space,
                    state=state,
                    qtable=self.qtable
                )
                
                # Exécuter l'action
                new_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                # Mettre à jour la Q-table
                self.qtable[state, action] = self.update(
                    state, action, reward, new_state
                )
                
                # Enregistrer les métriques
                callback.on_step(reward, info, done, timestep)
                
                # Incrémenter le timestep
                timestep += 1
                pbar.update(1)
                
                # Mettre à jour l'état
                state = new_state
        
        pbar.close()

    def save(self, path):
        """Sauvegarde la Q-table dans un fichier."""
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

class EpsilonGreedy:
    """
    Stratégie d'exploration epsilon-greedy pour la sélection d'actions. Le code pour cette classe a été
    adapté à partir de l'exemple d'entraînement Q-learning sur FrozenLake disponible sur le site de
    Gymnasium: https://gymnasium.farama.org/tutorials/training_agents/frozenlake_q_learning/
    """

    def __init__(self, epsilon):
        self.epsilon = epsilon

    def choose_action(self, action_space, state, qtable):
        """Choisit une action `a` dans l'état actuel du monde (s)."""
        # Premièrement, nous générons un nombre aléatoire
        explor_exploit_tradeoff = rng.uniform(0, 1)

        # Exploration
        if explor_exploit_tradeoff < self.epsilon:
            action = action_space.sample()

        # Exploitation (prendre la plus grande valeur Q pour cet état)
        else:
            # Casser les égalités aléatoirement
            # Trouver les indices où la valeur Q est égale à la valeur maximale
            # Choisir une action aléatoire parmi les indices où la valeur Q est maximale
            max_ids = np.where(qtable[state, :] == max(qtable[state, :]))[0]
            action = rng.choice(max_ids)
        return action
    
# ==================== CONFIGURATION ====================

# Paramètres de l'expérience
SUCCESS_RATE = 9 / 10.0  # Probabilité de ne pas glisser (slippery env)
MAX_EPISODE_STEPS = 60  # Durée maximale d'un épisode (nombre de pas)
TOTAL_TIMESTEPS = 100_000  # Nombre total de pas pour l'entraînement
ENV_TYPE = "10x10_custom"  # Type d'environnement: "8x8_default" ou "10x10_custom"
BASE_SEED = 42

# Paramètres Q-learning
LEARNING_RATE = 0.1  # Taux d'apprentissage (alpha)
GAMMA = 0.99  # Facteur de rabais (discount factor)
EPSILON = 0.1  # Probabilité d'exploration dans epsilon-greedy
QLEARNING_VERBOSE = 1  # Niveau de verbosité (1 = afficher les mises à jour)


# ==================== PIPELINE D'ENTRAÎNEMENT ====================

def main():
    """Entraîne une politique Q-learning et affiche les résultats."""
    
    # Créer l'environnement
    print(f"Création de l'environnement: {ENV_TYPE}")
    env = get_env_by_type(
        ENV_TYPE,
        max_episode_steps=MAX_EPISODE_STEPS,
        success_rate=SUCCESS_RATE,
        seed=BASE_SEED
    )
    
    # Récupérer la taille de l'espace d'états et d'actions
    state_size = env.observation_space.n
    action_size = env.action_space.n
    
    # Créer le modèle Q-learning
    print(f"\nInitialisation du modèle Q-learning...")
    print(f"  - Total timesteps: {TOTAL_TIMESTEPS:,}")
    print(f"  - Environnement: {ENV_TYPE}")
    
    learner = Qlearning(
        learning_rate=LEARNING_RATE,
        gamma=GAMMA,
        state_size=state_size,
        action_size=action_size,
        verbose=QLEARNING_VERBOSE
    )
    
    # Créer l'explorateur epsilon-greedy
    explorer = EpsilonGreedy(epsilon=EPSILON)
    
    # Créer la callback pour enregistrer les métriques
    callback = QLearningMetricsCallback(verbose=QLEARNING_VERBOSE)
    
    # Entraîner le modèle
    print(f"\nEntraînement du modèle Q-learning...")
    learner.learn(
        env=env,
        total_timesteps=TOTAL_TIMESTEPS,
        explorer=explorer,
        callback=callback
    )
    
    # Sauvegarder le modèle
    model_path = f"q_learning_{ENV_TYPE}_{BASE_SEED}.npy"
    learner.save(model_path)
    print(f"\nModèle sauvegardé: {model_path}")
    
    # Extraire et afficher la politique apprise
    print("\nExtraction de la politique apprise...")
    policy_grid = extract_policy_grid(env, learner)
    print()
    print_policy(policy_grid)
    
    # Fermer l'environnement
    env.close()
    print("\nEntraînement terminé.")

    # Plot metrics
    x_raw, x_ma, reward_mean, length_mean, fall_rate, timeout_rate, risk_mean, cum_falls = get_moving_averages_with_x(callback, window=30)

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
