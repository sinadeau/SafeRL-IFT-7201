

class MetricsCallback(BaseCallback):
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
            self.episode_risks.append(self.current_risk)
            self.episode_falls.append(fall)
            self.episode_successes.append(success)
            self.episode_timeouts.append(timeout)

            self.total_falls += fall
            self.cum_falls.append(self.total_falls)
            self.episode_end_timesteps.append(self.num_timesteps)

            self.current_reward = 0.0
            self.current_length = 0
            self.current_risk = 0

        return True