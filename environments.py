"""
Définition des environnements FrozenLake.
"""

import numpy as np
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.monitor import Monitor
from matplotlib.colors import LinearSegmentedColormap, Normalize
import matplotlib.pyplot as plt


def create_env(
    env_type: str,
    reward_type: str = "MDP", #"MDP" ou "CMDP" (Version safe-RL). Pour un CMDP la récompense de tomber dans un trou est de 0 pour dissocier le coût de la fonction de récompense.
    hole_cost: float = 1,
    max_episode_steps: int = 60,
    success_rate: float = 0.9,
    seed: int = 42
) -> gym.Env:
    """
    Factory pour créer un environnement FrozenLake selon le type spécifié.
    
    Args:
        env_type: Type d'environnement ("8x8_default", "10x10_custom" ou "16x16_random)
        reward_type: "MDP" ou "CMDP". Si "CMDP", la récompense de tomber dans un trou est de 0 pour le dissocier de la fonction de récompense
        hole_cost: coût utilisé pour si reward_type = "CMDP".
        max_episode_steps: Durée maximale d'un épisode (nombre de pas)
        success_rate: Probabilité de ne pas glisser (1.0 = déterministe)
        seed: graine pour la génération aléatoire
    
    Returns:
        gym.Env: Environnement encapsulé avec TimeLimit et Monitor
    """
    if env_type == "8x8_default":
        """
        Configuration inspirée de Zhu et al. (2025):
        - Grille 8x8
        - Trous (H) positionnés stratégiquement
        - Objectif (G) défini
        - reward_schedule = (1, -0.1, -0.01)  # (Goal, Hole, Frozen) pour un MDP
        (pour un CMDP la récompense de tomber dans un trou est de 0 pour le dissocier de la fonction de récompense)
        """

        if reward_type == "MDP":
            reward_schedule = (1, -0.1, -0.01)
        elif reward_type == "CMDP":
            reward_schedule = (1, 0, -0.01)
        else:
            raise ValueError(f"Type de reward schedule inconnu: {reward_type}")

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
        """
        Configuration adaptée pour tester le compromis entre longueur des chemins vers le but et le risque.

        - Grille 10x10
        - Configuration déduite empiriquement
        - reward_schedule = (1, -0.5, -0.01)  # (Goal, Hole, Frozen)
        (pour un CMDP la récompense de tomber dans un trou est de 0 pour le dissocier de la fonction de récompense)
        """

        if reward_type == "MDP":
            reward_schedule = (1, -0.5, -0.01)
        elif reward_type == "CMDP":
            reward_schedule = (1, 0, -0.01)
        else:
            raise ValueError(f"Type de reward schedule inconnu: {reward_type}")

        desc = [
            "SFFFFFFFFF",
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
    elif env_type == "16x16_random":
        """
        Configuration aléatoire pour tester la robustesse de l'agent à des environnements 
        plus grands et moins structurés.

        - Grille 16x16
        - Trous positionnés aléatoirement avec une densité de 15%
        - Objectif placé dans un coin opposé à la position de départ
        - reward_schedule = (1, -1, -0.01) # (Goal, Hole, Frozen)
        (pour un CMDP la récompense de tomber dans un trou est de 0 pour le dissocier de la fonction de récompense)
        """

        if reward_type == "MDP":
            reward_schedule = (1, -1, -0.01)
        elif reward_type == "CMDP":
            reward_schedule = (1, 0, -0.01)
        else:
            raise ValueError(f"Type de reward schedule inconnu: {reward_type}")

        # Définition de la grille
        np.random.seed(seed)
        size = 16
        desc = np.full((size, size), "F", dtype=str)

        # Placer les trous aléatoirement (15% de la grille)
        n_holes = int(0.15 * size * size)
        hole_positions = np.random.choice(size * size, n_holes, replace=False)
        for pos in hole_positions:
            row, col = divmod(pos, size)
            desc[row, col] = "H"

        # Placer le départ et l'objectif
        desc[0, 0] = "S"
        desc[size - 1, size - 1] = "G"

    else:
        raise ValueError(f"Type d'environnement inconnu: {env_type}")
    
    env = gym.make(
        "FrozenLake-v1",
        desc=desc,
        is_slippery=True,
        success_rate=success_rate,
        reward_schedule=reward_schedule,
        render_mode="rgb_array",
        max_episode_steps=max_episode_steps
    )
    env = FrozenLakeInfoWrapper(env, hole_cost = hole_cost)
    env = Monitor(env) #Pour log métriques standard dans tensorboard
    env.reset(seed=seed)
    
    return env

class FrozenLakeInfoWrapper(gym.Wrapper):
    """
        Wrapper pour enrichir l'environnement FrozenLake avec des informations
        supplémentaires utiles pour l'analyse des performances et de la sécurité.

        Ce wrapper ajoute dans le dictionnaire `info` des indicateurs permettant de :
        - distinguer les types de terminaison d'épisode (succès, chute, timeout)
        - associer à chaque état un niveau de risque basé sur la proximité des trous

        Le risque est défini comme une fonction du nombre de trous adjacents à la case courante. 
        Chaque trou adjacent contribue de manière additive au risque :
            risque = 2 x (nombre de trous adjacents)

        Informations ajoutées à `info` à chaque pas :
            - is_fall (bool) : True si l'agent est tombé dans un trou
            - is_success (bool) : True si l'agent a atteint l'objectif
            - is_timeout (bool) : True si l'épisode est terminé par dépassement du nombre de pas
            - risk (int) : niveau de risque de la case courante
            - cost(float) : coût de la case courante, utilisé seulement pour un CMDP

        Args:
            env (gym.Env): Environnement FrozenLake à encapsuler
            hole_cost: coût de tomber dans un trou, utilisé seulement pour un CMDP.

        Returns:
            gym.Env: Environnement enrichi avec des informations supplémentaires dans `info`
    """

    def __init__(self, env, hole_cost):
        super().__init__(env)

        base_env = env.unwrapped
        self.desc = base_env.desc
        self.desc_str = self.desc.astype(str)
        self.nrow, self.ncol = self.desc.shape
        self.hole_cost = hole_cost

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

        #Ajouter le coût dans info
        info["cost"] = self.hole_cost if info["is_fall"] else 0.0

        #Autre version possible du coût
        #info["cost"] = hole_cost if info["is_fall"] else beta * (info["risk"] / 6.0)

        return obs, reward, terminated, truncated, info


    def plot_risk_map(self, ax=None, title="Carte de risque de l'environnement", show_colorbar=True):
        # --- Colormap ---
        colors = [
            (0.0, "green"),
            (2/6, "yellow"),
            (4/6, "orange"),
            (1.0, "red")
        ]
        cmap = LinearSegmentedColormap.from_list("risk_cmap", colors)
        norm = Normalize(vmin=0, vmax=6)

        risk_map = self.risk_map
        desc = self.desc.astype(str)

        # Si aucun axe n'est fourni, on crée une figure
        created_fig = False
        if ax is None:
            fig, ax = plt.subplots(figsize=(5.5, 5))
            created_fig = True

        im = ax.imshow(risk_map, cmap=cmap, norm=norm)

        nrow, ncol = desc.shape

        for i in range(nrow):
            for j in range(ncol):
                tile = desc[i, j]

                if tile == "S":
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color="purple"))
                    ax.text(j, i, "S", ha="center", va="center",
                            color="white", fontsize=12, weight="bold")

                elif tile == "H":
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color="blue"))
                    ax.text(j, i, "H", ha="center", va="center",
                            color="white", fontsize=12, weight="bold")

                elif tile == "G":
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color="#40E0D0"))
                    ax.text(j, i, "G", ha="center", va="center",
                            color="black", fontsize=12, weight="bold")

                else:
                    val = risk_map[i, j]
                    ax.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=12)

        ax.set_title(title, fontsize=20)

        ax.set_xticks(range(ncol))
        ax.set_yticks(range(nrow))
        ax.tick_params(axis='both', labelsize=12, width=1.5, length=5)

        # colorbar seulement si demandé
        if show_colorbar:
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_ticks([0, 2, 4, 6])
            cbar.set_label("Risque", fontsize=20)

        if created_fig:
            plt.show()

        return im
    
    