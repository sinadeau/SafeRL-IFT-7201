import gymnasium as gym
from stable_baselines3 import PPO
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

#Simon
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
    success_rate=9.9/10.0,
    reward_schedule=(1, 0, 0)
)

#Notes sur environnement Custom Simon
#success_rate=3/4 passe par le plus long chemin
#success_rate=9.0/10.0 # passe par le chemin intermédiaire

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./8x8_tensorboard/")
model.learn(total_timesteps=5_000_000, tb_log_name="ppo_seed0")
model.save("ppo_8x8")
print("Modèle sauvegardé : ppo_8x8")

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
