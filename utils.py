"""
Fonctions utilitaires.
"""
import os
import re
import math
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import gymnasium as gym
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from stable_baselines3 import PPO
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure as sb3_configure
from environments import create_env


# ======== Fonctions de moyenne =========

def moving_average(x, window=100):
    """
    Calcule une moyenne glissante (moving average) sur une série de données.

    Args:
        x (array-like): Série de données (ex: rewards, longueurs d’épisodes, etc.)
        window (int): Taille de la fenêtre de lissage (nombre de points)

    Returns:
        np.ndarray:
            - Si len(x) >= window : série lissée de longueur len(x) - window + 1
            - Sinon : retourne la série originale (pas assez de points pour lisser)

    Notes:
        - Le mode "valid" garantit que seules les moyennes complètes sont conservées
          (pas de padding).
        - Chaque valeur retournée correspond à la moyenne des `window` points précédents.
    """

    x = np.asarray(x, dtype=float)

    if len(x) < window: # Pas assez de points pour lisser, retourner la série originale (mais ne risque pas d'arriver avec un nombre d'épisodes raisonnable)
        return x

    kernel = np.ones(window)
    sum_x = np.convolve(np.nan_to_num(x, nan=0.0), kernel, mode="valid")
    count = np.convolve(~np.isnan(x), kernel, mode="valid")
    out = sum_x / count
    out[count == 0] = np.nan  # Si aucune valeur valide dans la fenêtre, retourner NaN

    return out

def get_moving_averages_with_x(data, window=100):
    """
    Calcule les moyennes mobiles de plusieurs métriques d'entraînement RL
    et retourne les axes associés pour la visualisation.

    Cette fonction prend en entrée un dictionnaire contenant les métriques
    par épisode (récompense, longueur, chutes, succès, etc.), puis applique
    un lissage par moyenne mobile sur chacune d'elles.

    Elle retourne également deux axes :
    - x_raw : index des épisodes bruts
    - x_ma : index aligné avec les moyennes mobiles (corrigé du décalage dû au window)

    Args:
        data (dict): Dictionnaire contenant les métriques par épisode avec les clés :
            - "episode_rewards"
            - "episode_lengths"
            - "episode_lengths_given_success"
            - "episode_falls"
            - "episode_successes"
            - "episode_risks"
            - "episode_timeouts"
            - "cum_falls"
        window (int): Taille de la fenêtre pour la moyenne mobile

    Returns:
        tuple:
            x_raw (np.ndarray): Index des épisodes (1 à N)
            x_ma (np.ndarray): Index aligné avec les moyennes mobiles
            reward_mean (np.ndarray): Moyenne mobile des récompenses
            length_mean (np.ndarray): Moyenne mobile des longueurs d’épisodes
            length_given_success_mean (np.ndarray): Moyenne mobile des longueurs conditionnelles au succès
            fall_rate (np.ndarray): Moyenne mobile du taux de chute
            success_rate (np.ndarray): Moyenne mobile du taux de succès
            timeout_rate (np.ndarray): Moyenne mobile du taux de timeout
            risk_mean (np.ndarray): Moyenne mobile du risque
            cum_falls (np.ndarray): Nombre cumulatif de chutes (non lissé)
    """
    rewards = np.array(data["episode_rewards"])
    lengths = np.array(data["episode_lengths"])
    episode_lengths_given_success = np.array(data["episode_lengths_given_success"])
    falls = np.array(data["episode_falls"])
    successes = np.array(data["episode_successes"])
    risks = np.array(data["episode_risks"])
    timeouts = np.array(data["episode_timeouts"])
    cum_falls = np.array(data["cum_falls"])    
    #timesteps = np.array(data["episode_end_timesteps"])

    reward_mean = moving_average(rewards, window)
    length_mean = moving_average(lengths, window)
    length_given_success_mean = moving_average(episode_lengths_given_success, window)
    fall_rate = moving_average(falls, window)
    success_rate = moving_average(successes, window)
    timeout_rate = moving_average(timeouts, window)
    risk_mean = moving_average(risks, window)

    x_raw = np.arange(1, len(rewards) + 1)

    if len(rewards) < window:
        x_ma = x_raw
    else:
        x_ma = np.arange(window, len(rewards) + 1)

    return x_raw, x_ma, reward_mean, length_mean, length_given_success_mean,fall_rate, success_rate, timeout_rate, risk_mean, cum_falls

def average_smoothed_metrics_across_runs(callbacks_data_list, window=30):
    """
    Lisse les métriques de chaque répétition puis les moyenne sur toutes les répétitions.

    Args:
        callbacks_data_list: Liste de dicts de métriques (de load_callback_metrics_npz)
        window: Taille de la fenêtre de lissage

    Returns:
        dict: Métriques lissées et moyennées, avec les clés:
              x_ma, reward_mean, length_mean, fall_rate, timeout_rate, risk_mean,
              x_raw, cum_falls
    """
    smoothed, raw_cf = [], []
    for data in callbacks_data_list:
        x_raw, x_ma, rew, length, length_given_success, falls, success, timeout, risk, cf = \
            get_moving_averages_with_x(data, window=window)
        smoothed.append((x_ma, rew, length, length_given_success, falls, success, timeout, risk))
        raw_cf.append((x_raw, cf))

    # Aligner toutes les séries sur la longueur minimale (sécurité si les longueurs diffèrent)
    min_ma = min(len(r[1]) for r in smoothed)
    min_raw = min(len(r[1]) for r in raw_cf)

    return {
        "x_ma":         smoothed[0][0][:min_ma].astype(int),
        "reward_mean":  np.mean([r[1][:min_ma] for r in smoothed], axis=0),
        "length_mean":  np.mean([r[2][:min_ma] for r in smoothed], axis=0),
        "length_given_success_mean": np.nanmean([r[3][:min_ma] for r in smoothed], axis=0),
        "fall_rate":    np.mean([r[4][:min_ma] for r in smoothed], axis=0),
        "success_rate": np.mean([r[5][:min_ma] for r in smoothed], axis=0),
        "timeout_rate": np.mean([r[6][:min_ma] for r in smoothed], axis=0),
        "risk_mean":    np.mean([r[7][:min_ma] for r in smoothed], axis=0),

        "x_raw":        raw_cf[0][0][:min_raw].astype(int),
        "cum_falls":    np.mean([r[1][:min_raw] for r in raw_cf], axis=0),
    }


# ======== Calcul des métriques =========

class QLearningMetricsCallback:
    """
    Callback personnalisée pour suivre des métriques lors de l'entraînement Q-learning.
    
    Enregistre les mêmes métriques que MetricsCallback mais sans dépendre de Stable Baselines3.
    Cette classe accumule les informations pour chaque pas en mémoire et calcule les
    métriques agrégées lorsqu'un épisode se termine.

    Cette fonction a été générée par Claude Haiku 4.5, avec des ajustements pour correspondre aux
    métriques suivies par SB3MetricsCallback.
    """

    def __init__(self, verbose=0):
        self.verbose = verbose

        self.current_reward = 0.0
        self.current_length = 0
        self.current_risk = 0

        # Listes avec 1 valeur par épisode
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_lengths_given_success = []
        self.episode_falls = []
        self.episode_successes = []
        self.episode_timeouts = []
        self.episode_risks = []
        self.cum_falls = []
        self.episode_end_timesteps = []

        self.total_falls = 0
        self.num_timesteps = 0

    def __getitem__(self, key):
        """
        Permet l'accès aux attributs via la syntaxe de dictionnaire.
        
        Args:
            key: Nom de l'attribut à récupérer
            
        Returns:
            La valeur de l'attribut correspondant
        """
        return getattr(self, key)

    def on_step(self, reward: float, info: dict, done: bool, timestep: int) -> None:
        """
        Enregistre les informations d'un pas de simulation.
        
        Args:
            reward: Récompense obtenue à ce pas
            info: Dictionnaire d'information contenant is_fall, is_success, is_timeout, risk
            done: Booléen indiquant si l'épisode est terminé
            timestep: Numéro du timestep courant
        """
        self.current_reward += reward
        self.current_length += 1
        self.current_risk += info.get("risk", 0)
        self.num_timesteps = timestep

        if done:
            fall = int(info.get("is_fall", False))
            success = int(info.get("is_success", False))
            timeout = int(info.get("is_timeout", False))

            self.episode_rewards.append(self.current_reward)
            self.episode_lengths.append(self.current_length)
            self.episode_lengths_given_success.append(self.current_length if success else np.nan)
            self.episode_risks.append(self.current_risk)
            self.episode_falls.append(fall)
            self.episode_successes.append(success)
            self.episode_timeouts.append(timeout)

            self.total_falls += fall
            self.cum_falls.append(self.total_falls)
            self.episode_end_timesteps.append(self.num_timesteps)

            # --- Reset ---
            self.current_reward = 0.0
            self.current_length = 0
            self.current_risk = 0

class SB3MetricsCallback(BaseCallback):
    """
    Callback pour suivre des métriques par épisode durant l'entraînement.

    Cette classe accumule des informations à chaque pas de temps et,
    lorsque l'épisode se termine, enregistre des métriques agrégées par épisode :
    - reward total
    - longueur de l'épisode
    - risque cumulé
    - type de terminaison (chute, succès, timeout)
    - nombre cumulé de chutes

    Compatible avec un environnement non vectorisé (n_env = 1).
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)

        self.current_reward = 0.0
        self.current_length = 0
        self.current_risk = 0

        #listes avec 1 valeur par épisode
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_lengths_given_success = []
        self.episode_falls = []
        self.episode_successes = []
        self.episode_timeouts = []
        self.episode_risks = []
        self.cum_falls = []
        self.episode_end_timesteps = []

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
            self.episode_lengths_given_success.append(self.current_length if success else np.nan)
            self.episode_risks.append(self.current_risk)
            self.episode_falls.append(fall)
            self.episode_successes.append(success)
            self.episode_timeouts.append(timeout)

            self.total_falls += fall
            self.cum_falls.append(self.total_falls)
            self.episode_end_timesteps.append(self.num_timesteps)
            
            # --- Logs custom TensorBoard ---
            self.logger.record("custom/episode_reward", self.current_reward)
            self.logger.record("custom/episode_length", self.current_length)
            self.logger.record("custom/episode_length_given_success", self.current_length if success else np.nan)
            self.logger.record("custom/episode_risk", self.current_risk)
            self.logger.record("custom/fall", fall)
            self.logger.record("custom/success", success)
            self.logger.record("custom/timeout", timeout)
            self.logger.record("custom/cumulative_falls", self.total_falls)

            # Enregistrer les métriques dans TensorBoard à chaque épisode (step = indice d'épisode 0-indexé)
            self.logger.dump(step=len(self.episode_rewards) - 1)

            # --- Reset ---
            self.current_reward = 0.0
            self.current_length = 0
            self.current_risk = 0            

        return True

def save_callback_metrics_npz(callback, filepath):
    """
    Sauvegarde les métriques du callback dans un fichier .npz
    """

    np.savez(
        filepath,
        episode_rewards=np.array(callback.episode_rewards),
        episode_lengths=np.array(callback.episode_lengths),
        episode_lengths_given_success=np.array(callback.episode_lengths_given_success),
        episode_falls=np.array(callback.episode_falls),
        episode_successes=np.array(callback.episode_successes),
        episode_risks=np.array(callback.episode_risks),        
        episode_timeouts=np.array(callback.episode_timeouts),
        cum_falls=np.array(callback.cum_falls),       
        episode_end_timesteps=np.array(callback.episode_end_timesteps),
    )

def load_callback_metrics_npz(filepath):
    """
    Charge les métriques depuis un fichier .npz
    """
    data = np.load(filepath)
    return {key: data[key] for key in data.files}

def run_training_repetitions(train_fn, n_repetitions, base_seed, device):
    """
    Lance n_repetitions d'entraînement en parallèle via un pool de threads.

    Args:
        train_fn: Fonction prenant (seed) et retournant les données du callback (dict)
        n_repetitions: Nombre de répétitions
        base_seed: Graine de base ; la répétition i utilise base_seed + i
        device: objet torch.device

    Returns:
        list[dict]: Données de callback pour chaque répétition, dans l'ordre
    """
    seeds = [base_seed + i for i in range(n_repetitions)]

    if device.type == "cpu":  # parallèle CPU
        with ThreadPoolExecutor(max_workers=n_repetitions) as executor:
            futures = [executor.submit(train_fn, seed) for seed in seeds]
            return [f.result() for f in futures]
    elif device.type == "cuda": #GPU pas en parallèle
        results = []
        for seed in seeds:
            results.append(train_fn(seed))
        return results
        
       


# ======== Fonctions d'enregistrement =========

def log_run_metrics_to_tensorboard(callback_data, tensorboard_dir):
    """
    Écrit les métriques d'une seule répétition dans TensorBoard, une valeur par épisode.
    Garantit exactement len(episode_rewards) pas par métrique.

    Args:
        callback_data: Dict de métriques (de load_callback_metrics_npz)
        tensorboard_dir: Répertoire de sortie TensorBoard
    """
    # Nettoyer le dossier pour éviter de mélanger des séries de plusieurs exécutions.
    tb_path = Path(tensorboard_dir)
    shutil.rmtree(tb_path, ignore_errors=True)
    logger = sb3_configure(folder=str(tb_path), format_strings=["tensorboard"])
    n = len(callback_data["episode_rewards"])

    for i in range(n):
        logger.record("custom/episode_reward",   float(callback_data["episode_rewards"][i]))
        logger.record("custom/episode_length",   float(callback_data["episode_lengths"][i]))
        logger.record("custom/episode_length_given_success", float(callback_data["episode_lengths_given_success"][i]))
        logger.record("custom/episode_risk",     float(callback_data["episode_risks"][i]))
        logger.record("custom/fall",             float(callback_data["episode_falls"][i]))
        logger.record("custom/success",          float(callback_data["episode_successes"][i]))
        logger.record("custom/timeout",          float(callback_data["episode_timeouts"][i]))
        logger.record("custom/cumulative_falls", float(callback_data["cum_falls"][i]))
        logger.dump(step=i)
    logger.close()

def log_averaged_metrics_to_tensorboard(averaged_metrics, tensorboard_dir):
    """
    Enregistre les métriques lissées et moyennées dans TensorBoard.

    Args:
        averaged_metrics: Dict retourné par average_smoothed_metrics_across_runs
        tensorboard_dir: Répertoire de sortie pour les fichiers TensorBoard
    """
    # Nettoyer le dossier pour éviter de mélanger des séries de plusieurs exécutions.
    tb_path = Path(tensorboard_dir)
    shutil.rmtree(tb_path, ignore_errors=True)
    logger = sb3_configure(folder=str(tb_path), format_strings=["tensorboard"])

    # Métriques lissées, indexées par épisode lissé (x_ma)
    for i, ep in enumerate(averaged_metrics["x_ma"]):
        logger.record("averaged/reward_mean",  float(averaged_metrics["reward_mean"][i]))
        logger.record("averaged/length_mean",  float(averaged_metrics["length_mean"][i]))
        logger.record("averaged/length_given_success_mean", float(averaged_metrics["length_given_success_mean"][i]))
        logger.record("averaged/fall_rate",    float(averaged_metrics["fall_rate"][i]))
        logger.record("averaged/success_rate", float(averaged_metrics["success_rate"][i]))
        logger.record("averaged/timeout_rate", float(averaged_metrics["timeout_rate"][i]))
        logger.record("averaged/risk_mean",    float(averaged_metrics["risk_mean"][i]))
        logger.dump(step=int(ep))

    # Chutes cumulées, indexées sur tous les épisodes bruts (x_raw)
    for i, ep in enumerate(averaged_metrics["x_raw"]):
        logger.record("averaged/cum_falls", float(averaged_metrics["cum_falls"][i]))
        logger.dump(step=int(ep))

    logger.close()


# ======== Fonctions d'évaluation =========

def evaluate_policy_with_metrics(model, env, n_eval_episodes=100, deterministic=True):
    """
    Évalue une politique SB3 sur n épisodes et retourne des métriques customisées.

    Args:
        model: Modèle SB3 entraîné ayant une méthode predict()
        env: Environnement d'évaluation non vectorisé
        n_eval_episodes: Nombre d'épisodes à évaluer
        deterministic: Si True, utilise la politique déterministe ; sinon, stochastique

    Returns:
        dict: Métriques d'évaluation (mean_reward, mean_length, success_rate, fall_rate, etc.)
    """

    # Initialiser les listes pour accumuler les métriques par épisode
    episode_rewards = []
    episode_lengths = []
    episode_lengths_given_success = []    
    episode_falls = []
    episode_successes = []
    episode_timeouts = []
    episode_risks = []
    cum_falls = []

    total_falls = 0

    # Évaluer la politique sur n_eval_episodes épisodes
    for episode in range(n_eval_episodes):
        obs, info = env.reset()

        done = False
        truncated = False

        # Accumulateurs pour cet épisode
        current_reward = 0.0
        current_length = 0
        current_risk = 0.0

        # Simuler un épisode complet
        while not (done or truncated):
            # Prédire l'action selon la politique
            action, _states = model.predict(obs, deterministic=deterministic)
            
            if isinstance(action, np.ndarray):
                action = action.item()
            action = int(action)
                        
            obs, reward, done, truncated, info = env.step(action)

            current_reward += reward
            current_length += 1
            current_risk += info.get("risk", 0)

        # Extraire les informations de terminaison de l'épisode
        fall = int(info.get("is_fall", False))
        success = int(info.get("is_success", False))
        timeout = int(info.get("is_timeout", False))

        # Enregistrer les métriques de l'épisode
        episode_rewards.append(current_reward)
        episode_lengths.append(current_length)
        episode_lengths_given_success.append(current_length if success else np.nan)  # NaN pour les épisodes non réussis
        episode_risks.append(current_risk)
        episode_falls.append(fall)
        episode_successes.append(success)
        episode_timeouts.append(timeout)

        total_falls += fall
        cum_falls.append(total_falls)

    # Calculer les métriques moyennées sur tous les épisodes
    results = {
        "mean_reward": np.mean(episode_rewards),
        "mean_length": np.mean(episode_lengths),
        "mean_length_given_success": np.nanmean(episode_lengths_given_success),  # Moyenne en ignorant les NaN        
        "mean_risk": np.mean(episode_risks),
        "fall_rate": np.mean(episode_falls),
        "success_rate": np.mean(episode_successes),
        "timeout_rate": np.mean(episode_timeouts),
        "total_falls": total_falls,
    }

    return results

def average_eval_metrics_across_runs(eval_data_list):
    """
    Calcule les métriques d'évaluation moyennées et leurs écarts-types sur plusieurs répétitions.

    Cette fonction agrège les résultats d'évaluations répétées d'une même politique pour obtenir
    des statistiques robustes. Elle ignore les valeurs NaN (ex. length_given_success pour les
    épisodes échoués) lors du calcul des moyennes.

    Args:
        eval_data_list: Liste de dicts de métriques d'évaluation (chaque dict issu d'une évaluation)

    Returns:
        tuple: (moyennes, écarts-types)
            - moyennes: dict avec les métriques d'évaluation moyennées
            - écarts-types: dict avec les écarts-types correspondants (clés avec suffixe "_std")
    """

    keys = eval_data_list[0].keys()

    averaged = {}
    std = {}
    # Agréger chaque métrique en calculant sa moyenne et son écart-type
    for k in keys:
        values = [d[k] for d in eval_data_list]
        values = np.array(values, dtype=float)
        # nanmean/nanstd ignore les valeurs NaN (ex. length_given_success pour épisodes échoués)
        averaged[k] = np.nanmean(values)
        std[f"{k}_std"] = np.nanstd(values, ddof=1) # Diviser par n-1 pour l'écart-type d'un échantillon
    return averaged, std

def eval_metrics_to_dataframe(averaged, std):
    """
    Convertit les métriques d'évaluation moyennées en un DataFrame pandas pour faciliter
    l'affichage et la sauvegarde.

    Args:
        averaged: Dict des métriques moyennées (issus de average_eval_metrics_across_runs)
        std: Dict des écarts-types correspondants

    Returns:
        pd.DataFrame: DataFrame avec colonnes [metric, mean, std] pour chaque métrique
    """
    # Construire une liste de dictionnaires (une ligne par métrique)
    rows = []
    for k in averaged.keys():
        rows.append({
            "metric": k,
            "mean": averaged[k],
            "std": std[f"{k}_std"]
        })
    # Créer le DataFrame à partir des lignes
    df = pd.DataFrame(rows)
    return df


# ======== Fonctions d'extraction et de visualisation de la politique apprise =========
# Une partie du code des fonctons a été générée par ChatGPT 5.3 et 5.4

# Correspondance actions -> symboles de direction
ARROW_MAP = {
    0: "←",  # Gauche
    1: "↓",  # Bas
    2: "→",  # Droite
    3: "↑"   # Haut
}

def extract_policy_grid(env: gym.Env, model: BasePolicy) -> np.ndarray:
    """
    Extrait la politique apprise par le modèle sous forme de grille.
    
    Pour chaque état de l'environnement FrozenLake, récupère l'action préférentielle
    du modèle et la représente par un symbole de direction. Les cases spéciales
    (trous, objectif) restent identifiées.
    
    Args:
        env: Environnement FrozenLake encapsulé
        model: Modèle entraîné (ex. PPO, DQN) ayant une méthode predict()
        
    Returns:
        np.ndarray: Grille 2D contenant les symboles de politique ou d'état spécial
    """
    # Récupérer la description de la carte
    desc = env.unwrapped.desc.astype(str)
    nrow, ncol = desc.shape
    policy_grid = np.empty((nrow, ncol), dtype=object)
    
    # Parcourir tous les états
    for s in range(nrow * ncol):
        row, col = divmod(s, ncol)
        cell = desc[row, col]
        
        # Garder les cases spéciales
        if cell in ["H", "G"]:
            policy_grid[row, col] = cell
        else:
            # Obtenir l'action préférentielle du modèle
            action, _ = model.predict(s, deterministic=True)
            policy_grid[row, col] = ARROW_MAP[int(action)]
    
    return policy_grid

def extract_policy_grid_from_q_table(env, q_table):
    """
    Extrait la politique optimale à partir d'une Q-table sous forme de grille.

    Pour chaque état de l'environnement FrozenLake, sélectionne l'action
    ayant la plus grande valeur dans la Q-table (argmax) et la représente
    par un symbole de direction. Les cases spéciales (trous, objectif, départ)
    sont conservées telles quelles.

    Args:
        env: Environnement FrozenLake encapsulé
        q_table (np.ndarray): Tableau de taille (n_states, n_actions)
                             contenant les valeurs Q

    Returns:
        list[list[str]]: Grille 2D contenant les symboles de politique
                         ou les cases spéciales ("H", "G", "S")
    """
    base_env = env.unwrapped
    desc = base_env.desc
    nrow, ncol = desc.shape

    action_map = {
        0: "←",
        1: "↓",
        2: "→",
        3: "↑"
    }

    policy_grid = []

    for i in range(nrow):
        row = []
        for j in range(ncol):
            state = i * ncol + j
            tile = desc[i, j].decode() if isinstance(desc[i, j], bytes) else str(desc[i, j])

            if tile in ["H", "G", "S"]:
                row.append(tile)
            else:
                best_action = np.argmax(q_table[state])
                row.append(action_map[best_action])
        policy_grid.append(row)

    return policy_grid

def plot_policy(env, policy_grid, ax=None, title=None):
    """
    Affiche une politique sous forme de grille visuelle.

    Chaque cellule de la grille correspond à un état de l'environnement :
    - Les flèches indiquent l'action choisie par la politique
    - Les cases spéciales sont colorées :
        H (trou) -> bleu
        G (objectif) -> vert
        S (départ) -> jaune

    Args:
        env: Environnement FrozenLake encapsulé
        policy_grid: Grille 2D contenant la politique (flèches ou symboles)
        ax (matplotlib.axes.Axes, optional): Axe sur lequel dessiner la figure.
                                             Si None, une nouvelle figure est créée.
        title (str, optional): Titre à afficher au-dessus de la grille

    Returns:
        None
    """
    base_env = env.unwrapped
    desc = base_env.desc
    nrow, ncol = desc.shape

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    for i in range(nrow):
        for j in range(ncol):
            tile = desc[i, j].decode() if isinstance(desc[i, j], bytes) else str(desc[i, j])
            action = str(policy_grid[i][j])

            if tile == "H":
                color = "lightblue"
            elif tile == "G":
                color = "lightgreen"
            elif tile == "S":
                color = "lightyellow"
            else:
                color = "white"

            ax.add_patch(Rectangle((j, i), 1, 1, facecolor=color, edgecolor="black"))

            x = j + 0.5
            y = i + 0.5

            if tile in ["H", "G", "S"]:
                ax.text(x, y, tile, ha="center", va="center",
                        fontsize=12, fontweight="bold")
            else:
                ax.text(x, y, action, ha="center", va="center",
                        fontsize=12)

    ax.set_xlim(0, ncol)
    ax.set_ylim(nrow, 0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    if title is not None:
        ax.set_title(title)

def plot_all_PPO_policies_from_folder(env_type, folder="./ppo_10x10_custom", ncols=2):
    """
    Charge plusieurs modèles PPO depuis un dossier et affiche leurs politiques
    sous forme de grille de figures.

    Les modèles doivent être sauvegardés au format .zip avec un nom du type
    "model_X.zip". Les politiques sont extraites et affichées dans une figure
    composée de plusieurs sous-graphiques.

    Args:
        env_type: Type d'environnement ("8x8_default", "10x10_custom" ou "16x16_random)
        folder (str): Chemin vers le dossier contenant les modèles PPO
        ncols (int): Nombre de colonnes dans la grille de sous-figures

    Returns:
        None
    """
    model_paths = glob.glob(os.path.join(folder, "model_*.zip"))

    if len(model_paths) == 0:
        raise ValueError(f"Aucun fichier model_*.zip trouvé dans {folder}")

    def extract_model_number(path):
        match = re.search(r"model_(\d+)\.zip$", os.path.basename(path))
        return int(match.group(1)) if match else float("inf")

    model_paths = sorted(model_paths, key=extract_model_number)

    n_models = len(model_paths)
    nrows = math.ceil(n_models / ncols)

    cell_size = 3
    fig_width = ncols * cell_size
    fig_height = nrows * cell_size + 0.5 * nrows

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(fig_width, fig_height)
    )

    axes = np.array(axes).reshape(-1)

    for ax, model_path in zip(axes, model_paths):

        #Créer l'environnement
        seed = extract_model_number(model_path)
        env = create_env(env_type=env_type, seed=seed)

        model = PPO.load(model_path, env=env)
        policy_grid = extract_policy_grid(env, model)

        model_name = os.path.basename(model_path).replace(".zip", "")
        plot_policy(env, policy_grid, ax=ax, title=model_name)

    for ax in axes[n_models:]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(rf".\Figures\PPO_policies_{env_type}.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_all_q_learning_policies_from_folder(env_type, folder="./q_learning_10x10_custom", ncols=2):
    """
    Charge plusieurs Q-tables sauvegardées au format .npy et affiche les
    politiques correspondantes sous forme de grille de figures.

    Pour chaque fichier, la politique est reconstruite en prenant l'action
    de valeur maximale (argmax) pour chaque état.

    Args:
        env_type: Type d'environnement ("8x8_default", "10x10_custom" ou "16x16_random)
        folder (str): Chemin vers le dossier contenant les fichiers .npy
        ncols (int): Nombre de colonnes dans la grille de sous-figures

    Returns:
        None
    """     
    
    
    model_paths = glob.glob(os.path.join(folder, "*.npy"))

    if len(model_paths) == 0:
        raise ValueError(f"Aucun fichier .npy trouvé dans {folder}")

    def extract_model_number(path):
        match = re.search(r"(\d+)", os.path.basename(path))
        return int(match.group(1)) if match else float("inf")

    model_paths = sorted(model_paths, key=extract_model_number)

    n_models = len(model_paths)
    nrows = math.ceil(n_models / ncols)
    
    cell_size = 3
    fig_width = ncols * cell_size
    fig_height = nrows * cell_size + 0.5 * nrows 

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(fig_width, fig_height)
    )

    axes = np.array(axes).reshape(-1)

    for ax, model_path in zip(axes, model_paths):

        #Créer l'environnement
        seed = extract_model_number(model_path)
        env = create_env(env_type=env_type, seed=seed)

        q_table = np.load(model_path)
        policy_grid = extract_policy_grid_from_q_table(env, q_table)

        model_name = os.path.basename(model_path).replace(".npy", "")
        plot_policy(env, policy_grid, ax=ax, title=model_name)

    for ax in axes[n_models:]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(rf".\Figures\Q_learning_policies_{env_type}.png", dpi=300, bbox_inches='tight')
    plt.show()