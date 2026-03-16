import gymnasium as gym
from stable_baselines3 import PPO, DQN
import numpy as np
import matplotlib.pyplot as plt

#--- Default environment---
#env = gym.make(
#    'FrozenLake-v1',
#    map_name="8x8",
#    is_slippery=True,
#    success_rate=2.0/3.0,
#    reward_schedule=(1, 0, 0)       
#)

#---Custom environments---

#Simon 8 x 8 
desc= [
        "SFFFFFFF",
        "FFFFFFFF",
        "HFHFFFFF",
        "HFFFHHFF",
        "HFHFFFFF",
        "HFFFHFFF",
        "FFFFFFFF",
        "GFFFFFFF"
    ] 

#Notes sur environnement Custom Simon 8x8
#success_rate=3/4 passe par le plus long chemin
#success_rate=9.0/10.0 # passe par le chemin intermédiaire
#success_rate=9.9/10.0 # passe par le chemin le plus court et plus risqué

#Simon 10 x 10
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


#Héloïse
#desc= [
#        "SFFFFFHF",
#        "FFFFHFFH",
#        "HFFFFHFF",
#        "FFFHFHFF",
#        "FFFFFHFF",
#        "FFFHFHFH",
#        "FHFFFFFF",
#        "FFFFHFFG"
#    ]

#Héloïse 2
#desc= [
#    "SFFHF",
#    "HFFFH",
#    "HHFFF",
#    "GFFFF"
#]

env = gym.make(
    "FrozenLake-v1",
    desc= desc,
    is_slippery=True,
    success_rate=9/10.0,
    #reward_schedule=(1, 0, 0)
    reward_schedule=(1, -1, -0.01) #pénalité de -0.01 par pas de temps.
)

#---Entraînement---

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./10x10_tensorboard/")
model.learn(total_timesteps=1_000_000, tb_log_name="ppo_pen_minus1")

#model = DQN("MlpPolicy", env, verbose=1, tensorboard_log="./8x8_tensorboard/")
#model.learn(total_timesteps=5_000_000, log_interval=4, tb_log_name="DQN_seed0")

model.save("ppo_10x10")
print("Modèle sauvegardé : ppo_10x10")

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
