"""
Fonctions utilitaires.
"""

import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure as sb3_configure

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

def print_policy(policy_grid: np.ndarray) -> None:
    """
    Affiche la politique apprise dans la console.
    
    Args:
        policy_grid: Grille de politique obtenue via extract_policy_grid()
    """
    print("Politique apprise:")
    for row in policy_grid:
        print("  ".join(row))

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
    #Ancienne version simple (mais qui ne gère pas les NaN et peut retourner des valeurs plus courtes que x):
    #if len(x) < window:
    #    return x
    #return np.convolve(x, np.ones(window)/window, mode="valid")

    x = np.asarray(x, dtype=float)

    if len(x) < window: # Pas assez de points pour lisser, retourner la série originale (mais ne risque pas d'arriver avec un nombre d'épisodes raisonnable)
        return x

    kernel = np.ones(window)
    sum_x = np.convolve(np.nan_to_num(x, nan=0.0), kernel, mode="valid")
    count = np.convolve(~np.isnan(x), kernel, mode="valid")
    out = sum_x / count
    out[count == 0] = np.nan  # Si aucune valeur valide dans la fenêtre, retourner NaN

    return out

def get_moving_averages_with_x(data, window=30):
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

def compute_rollout_metrics(callback: 'QLearningMetricsCallback') -> dict:
    """
    Calcule les métriques rollout moyennes à partir des données d'entraînement.
    
    Args:
        callback: Instance de QLearningMetricsCallback contenant les métriques collectées
        
    Returns:
        dict: Dictionnaire contenant les métriques rollout moyennes:
            - ep_len_mean: Longueur moyenne des épisodes
            - ep_rew_mean: Récompense moyenne des épisodes
            - success_rate: Taux de succès (proportion d'épisodes avec succès)
    """
    if not callback.episode_lengths:
        return {"ep_len_mean": 0.0, "ep_rew_mean": 0.0, "success_rate": 0.0}
    
    ep_len_mean = np.mean(callback.episode_lengths)
    ep_rew_mean = np.mean(callback.episode_rewards)
    success_rate = np.mean(callback.episode_successes)
    
    return {
        "ep_len_mean": ep_len_mean,
        "ep_rew_mean": ep_rew_mean,
        "success_rate": success_rate
    }

class QLearningMetricsCallback:
    """
    Callback personnalisée pour suivre des métriques lors de l'entraînement Q-learning.
    
    Enregistre les mêmes métriques que MetricsCallback mais sans dépendre de Stable Baselines3.
    Cette classe accumule les informations pour chaque pas en mémoire et calcule les
    métriques agrégées lorsqu'un épisode se termine.
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

def run_training_repetitions(train_fn, n_repetitions, base_seed):
    """
    Lance n_repetitions d'entraînement en parallèle via un pool de threads.

    Args:
        train_fn: Fonction prenant (seed) et retournant les données du callback (dict)
        n_repetitions: Nombre de répétitions
        base_seed: Graine de base ; la répétition i utilise base_seed + i

    Returns:
        list[dict]: Données de callback pour chaque répétition, dans l'ordre
    """
    seeds = [base_seed + i for i in range(n_repetitions)]
    with ThreadPoolExecutor(max_workers=n_repetitions) as executor:
        futures = [executor.submit(train_fn, seed) for seed in seeds]
        return [f.result() for f in futures]

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