import gymnasium as gym
from stable_baselines3 import PPO
import numpy as np
import matplotlib.pyplot as plt

#env = gym.make(
#    'FrozenLake-v1',
#    map_name="4x4",
#    is_slippery=True,
#    success_rate=2.0/3.0,
#    reward_schedule=(1, 0, 0),
#    #render_mode="human"       
#)

#env = gym.make(
#    "FrozenLake-v1",
#    desc= [
#        "SFFFFFFF",
#        "FFFFFFFF",
#        "HFHFFFFF",
#        "HFFFHHFF",
#        "HFHFFFFF",
#        "HFFFHFFF",
#        "FFFFFFFF",
#        "GFFFFFFF"
#    ],
#    is_slippery=True,
#    success_rate=1.0/3.0,
#    reward_schedule=(1, 0, 0),
#    render_mode="human"
#)

#Héloïse
env = gym.make(
    "FrozenLake-v1",
    desc= [
        "SFFFFFHF",
        "FFFFHFFH",
        "HFFFFHFF",
        "FFFHFHFF",
        "FFFFFHFF",
        "FFFHFHFH",
        "FHFFFFFF",
        "FFFFHFFG"
    ],
    is_slippery=True,
    success_rate=3.0/3.0,
    reward_schedule=(1, 0, 0),
    #render_mode="human"
)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1000000)

#---Montrer une trajectoire avec la politique apprise (render human)
#state, info = env.reset()

#while True:
#    action, _states = model.predict(state)
#    state, reward, terminated, truncated, info = env.step(action)
#    env.render()
#
#    if terminated or truncated:
#        break

#env.close()

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

# Pour FrozenLake 4x4 standard, les observations sont des états discrets 0..15
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