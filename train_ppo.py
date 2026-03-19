import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from gymnasium.wrappers import TimeLimit
import time
import numpy as np
import matplotlib.pyplot as plt

#---Fonctions utilitaires---

class FrozenLakeInfoWrapper(gym.Wrapper):
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

#Fabrique d'environnement
def make_frozen_env(desc, reward_schedule, success_rate, max_episode_steps, seed):
    def _init():
        env = gym.make(
            "FrozenLake-v1",
            desc=desc,
            is_slippery=True,
            success_rate=success_rate,
            reward_schedule=reward_schedule,
        )
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
        env = FrozenLakeInfoWrapper(env)
        env.reset(seed=seed)
        return env
    return _init

class MetricsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

        self.current_reward = 0.0
        self.current_length = 0
        self.current_risk = 0

        #listes avec 1 valeur par épisode
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_falls = []
        self.episode_successes = []
        self.episode_timeouts = []
        self.episode_risks = []
        self.cum_falls = []
        self.episode_end_timesteps = []

        self.n_episodes = 0
        self.total_falls = 0

    def _on_step(self) -> bool:
        reward = float(self.locals["rewards"][0])
        done = bool(self.locals["dones"][0])
        info = self.locals["infos"][0]

        self.current_reward += reward
        self.current_length += 1
        self.current_risk += info.get("risk", 0)

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

            self.n_episodes += 1
            self.total_falls += fall
            self.cum_falls.append(self.total_falls)
            self.episode_end_timesteps.append(self.num_timesteps)

            self.current_reward = 0.0
            self.current_length = 0
            self.current_risk = 0

        return True

#--- Paramètres de l'expérience ---
success_rate=9/10.0 # probabilité de ne pas glisser
max_episode_steps = 60 # Limiter la durée d'un épisode à 60 pas
#total_timesteps = 10_000_000 #Nombre de pas de temps totaux pour l'entraînement
total_timesteps = 5_000
env_type = "10x10_custom" #Environnement (cas d'étude) choisi "8x8_default" ou "10x10_custom"
seed = 42

#---Définition de l'environnement

#Pour environement par défaut 8x8
if env_type == "8x8_default":
    reward_schedule=(1, -0.1, -0.01) #Reach Goal, Reach Hole, Reach Frozen (idem à Zhu et. al 2025) 

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

elif env_type == "10x10_custom":
    #Pour environement custom 10x10
    reward_schedule=(1, -0.5, -0.01) #Reach Goal, Reach Hole, Reach Frozen 

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

###version non-vectorisée
n_envs = 1 ##############################

env = make_frozen_env(
    desc=desc,
    reward_schedule=reward_schedule,
    success_rate=success_rate,
    max_episode_steps=max_episode_steps,
    seed=seed
)()

#env = gym.make(
#    "FrozenLake-v1",
#    desc=desc,
#    is_slippery=True,
#    success_rate=success_rate,
#    reward_schedule=reward_schedule
#)

#env = TimeLimit(env, max_episode_steps=max_episode_steps)
#env = Monitor(env)



#---Entraînement---
start_time = time.time()
callback = MetricsCallback()
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=f"./{env_type}_tensorboard/", seed=seed)
model.learn(total_timesteps=total_timesteps, tb_log_name=f"ppo_seed_{seed}", callback=callback)
end_time = time.time()
print(f"Temps d'exécution : {end_time - start_time} secondes")

model.save(f"ppo_{env_type}_seed_{seed}")
print(f"Modèle sauvegardé : ppo_{env_type}_seed_{seed}")

#--- Montrer la politique apprise ---
# Récupérer la carte FrozenLake
desc = env.unwrapped.desc.astype(str)

# Correspondance actions -> flèches
arrow_map = {
    0: "←",
    1: "↓",
    2: "→",
    3: "↑"
}

nrow, ncol = desc.shape
policy_grid = np.empty((nrow, ncol), dtype=object)

# Pour FrozenLake 8x8, les observations sont des états discrets 0..63
for s in range(nrow * ncol):
    r, c = divmod(s, ncol)
    cell = desc[r, c]

    # Garder les cases spéciales telles quelles
    if cell in ["H", "G"]:
        policy_grid[r, c] = cell
    else:
        action, _ = model.predict(s, deterministic=True)
        policy_grid[r, c] = arrow_map[int(action)]

print("Politique apprise :")
for row in policy_grid:
    print("  ".join(row))

env.close()


#Contruire les métriques---
rewards = np.array(callback.episode_rewards)
lengths = np.array(callback.episode_lengths)
falls = np.array(callback.episode_falls)
risks = np.array(callback.episode_risks)
successes = np.array(callback.episode_successes)
timeouts = np.array(callback.episode_timeouts)
cum_falls = np.array(callback.cum_falls)
timesteps = np.array(callback.episode_end_timesteps)

def moving_average(x, window=20):
    if len(x) < window:
        return x
    return np.convolve(x, np.ones(window)/window, mode="valid")

window = 30
reward_mean = moving_average(rewards, window=window)
length_mean = moving_average(lengths, window=window)
fall_rate = moving_average(falls, window=window)
timeout_rate = moving_average(timeouts, window=window)
risk_mean = moving_average(risks, window=window)

n_episodes_raw = np.arange(1, len(rewards) + 1)
n_episodes_ma = np.arange(window, len(rewards) + 1)

#def moving_average_with_x(x, window=20):
#    if len(x) < window:
#        return np.arange(1, len(x) + 1), x
#    
#    y = np.convolve(x, np.ones(window)/window, mode="valid")
#    x_axis = np.arange(window, len(x) + 1)
#    
#    return x_axis, y

fig, axs = plt.subplots(2, 3, figsize=(12, 8))

axs[0, 0].plot(n_episodes_ma, reward_mean)
axs[0, 0].set_title("Reward moyen par épisode")
axs[0, 0].set_xlabel("Épisodes")

axs[0, 1].plot(n_episodes_ma, length_mean)
axs[0, 1].set_title("Longueur moyenne de l'épisode")
axs[0, 1].set_xlabel("Épisodes")

axs[0, 2].plot(n_episodes_ma, fall_rate)
axs[0, 2].set_title("Taux de chute")
axs[0, 2].set_xlabel("Épisodes")

axs[1, 0].plot(n_episodes_raw, cum_falls)
axs[1, 0].set_title("Nombre cumulé de chutes")
axs[1, 0].set_xlabel("Épisodes")

axs[1, 1].plot(n_episodes_ma, risk_mean)
axs[1, 1].set_title("Risque moyen par épisode")
axs[1, 1].set_xlabel("Épisodes")

axs[1, 2].plot(n_episodes_ma, timeout_rate)
axs[1, 2].set_title("Taux de timeout")
axs[1, 2].set_xlabel("Épisodes")


plt.tight_layout()
plt.show()