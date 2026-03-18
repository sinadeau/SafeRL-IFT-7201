import gymnasium as gym
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import TimeLimit
import numpy as np
import matplotlib.pyplot as plt

#--- Paramètres de l'expérience ---
success_rate=9/10.0 # probabilité de ne pas glisser
max_episode_steps = 60 # Limiter la durée d'un épisode à 60 pas
total_timesteps = 500_000 #Nombre de pas de temps totaux pour l'entraînement
env_type = "8x8_default" #Environnement (cas d'étude) choisi "8x8_default" ou "10x10_custom"

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

env = gym.make(
    "FrozenLake-v1",
    desc=desc,
    is_slippery=True,
    success_rate=success_rate,
    reward_schedule=reward_schedule
)

env = TimeLimit(env, max_episode_steps=max_episode_steps)
env = Monitor(env)

#---Entraînement---
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=f"./{env_type}_tensorboard/")
model.learn(total_timesteps=total_timesteps, tb_log_name=f"ppo_{env_type}")

model.save(f"ppo_{env_type}")
print(f"Modèle sauvegardé : ppo_{env_type}")

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
