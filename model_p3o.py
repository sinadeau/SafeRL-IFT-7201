"""
Implémentation de l'algorithme P3O (Penalized Proximal Policy Optimization) basé sur stable-baselines3.

P3O étend PPO en ajoutant une pénalité pour les violations de contraintes de coût, 
inspiré de l'article "Penalized Proximal Policy Optimization for Safe Reinforcement Learning"
(Zhang et al., 2022, https://arxiv.org/pdf/2205.11814.pdf).

Le code est basé sur l'implémentation PPO de stable-baselines3:
https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/ppo/ppo.py

Les modifications par rapport à PPO standard sont identifiées avec le commentaire: [P3O]
"""

from collections import deque
from typing import Any, NamedTuple, Optional
from collections.abc import Generator

import numpy as np
import torch as th
import torch.nn as nn
from torch.nn import functional as F
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from stable_baselines3.common.utils import obs_as_tensor, explained_variance
from stable_baselines3.common.vec_env import VecEnv, VecNormalize


# ==================== [P3O] BUFFER AVEC COÛTS ====================

class CostRolloutBufferSamples(NamedTuple):
    """Échantillons (mini-batch) du buffer incluant les données de coût."""
    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    # [P3O] Champs additionnels pour les coûts
    cost_advantages: th.Tensor
    cost_returns: th.Tensor
    old_cost_values: th.Tensor


class CostRolloutBuffer(RolloutBuffer):
    """
    Extension du RolloutBuffer de SB3 pour stocker les coûts et avantages de coût.
    
    [P3O] Ce buffer ajoute le suivi des coûts, valeurs de coût, avantages de coût
    et retours de coût nécessaires pour la pénalité P3O.
    """

    costs: np.ndarray
    cost_values: np.ndarray
    cost_advantages: np.ndarray
    cost_returns: np.ndarray

    def reset(self) -> None:
        super().reset()
        # [P3O] Tableaux additionnels pour les coûts
        self.costs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.cost_values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.cost_advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.cost_returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: th.Tensor,
        log_prob: th.Tensor,
        cost: np.ndarray = None,           # [P3O]
        cost_value: th.Tensor = None,      # [P3O]
    ) -> None:
        # [P3O] Stocker les coûts avant l'appel parent (qui incrémente self.pos)
        if cost is not None:
            self.costs[self.pos] = np.array(cost)
        if cost_value is not None:
            self.cost_values[self.pos] = cost_value.clone().cpu().numpy().flatten()

        super().add(obs, action, reward, episode_start, value, log_prob)

    def compute_returns_and_advantage(self, last_values: th.Tensor, dones: np.ndarray,
                                       last_cost_values: th.Tensor = None) -> None:
        """Calcule les retours GAE (Generalized Advantage Estimation) pour les récompenses ET les coûts."""
        super().compute_returns_and_advantage(last_values, dones) #Retours GAE pour les récompenses

        # [P3O] GAE pour les coûts (mêmes gamma/lambda que pour les récompenses)
        if last_cost_values is not None:
            last_cv = last_cost_values.clone().cpu().numpy().flatten() #estimation de la valeur de coût au dernier pas de temps précédent du buffer (un pour chaque environnement)
            last_gae_lam = 0
            for step in reversed(range(self.buffer_size)):
                if step == self.buffer_size - 1:
                    next_non_terminal = 1.0 - dones.astype(np.float32)
                    next_values = last_cv
                else:
                    next_non_terminal = 1.0 - self.episode_starts[step + 1]
                    next_values = self.cost_values[step + 1]
                delta = self.costs[step] + self.gamma * next_values * next_non_terminal - self.cost_values[step]
                last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
                self.cost_advantages[step] = last_gae_lam
            self.cost_returns = self.cost_advantages + self.cost_values

    def get(self, batch_size: Optional[int] = None) -> Generator[CostRolloutBufferSamples, None, None]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)

        if not self.generator_ready:
            # [P3O] Ajouter les tableaux de coût au processus de mise à plat
            _tensor_names = [
                "observations", "actions", "values", "log_probs",
                "advantages", "returns",
                "cost_advantages", "cost_returns", "cost_values",
            ]
            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(
        self,
        batch_inds: np.ndarray,
        env: Optional[VecNormalize] = None,
    ) -> CostRolloutBufferSamples:
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
            # [P3O]
            self.cost_advantages[batch_inds].flatten(),
            self.cost_returns[batch_inds].flatten(),
            self.cost_values[batch_inds].flatten(),
        )
        return CostRolloutBufferSamples(*tuple(map(self.to_torch, data)))


# ==================== [P3O] CRITIQUE DE COÛT ====================

class CostCritic(nn.Module):
    """
    Réseau de critique de coût indépendant.
    
    [P3O] Ce réseau estime V_c(s), la valeur de coût attendue pour un état donné.
    Il est structuré comme un wrapper autour de la politique ActorCritic existante :
    il réutilise l'extracteur de features de la politique, mais possède son propre MLP et tête de valeur.
    """

    def __init__(self, features_dim: int, net_arch: list[int], activation_fn: type[nn.Module]):
        super().__init__()
        layers: list[nn.Module] = []
        last_dim = features_dim
        for layer_dim in net_arch:
            layers.append(nn.Linear(last_dim, layer_dim))
            layers.append(activation_fn())
            last_dim = layer_dim
        self.mlp = nn.Sequential(*layers)
        self.value_head = nn.Linear(last_dim, 1)

    def forward(self, features: th.Tensor) -> th.Tensor:
        return self.value_head(self.mlp(features))


# ==================== [P3O] ALGORITHME P3O ====================

class P3O(PPO):
    """
    Penalized Proximal Policy Optimization (P3O).

    Étend PPO avec une pénalité de coût pour l'apprentissage par renforcement sûr (Safe RL).
    La perte totale de l'acteur est : L_reward + kappa * ReLU(surr_cost_adv + Jc)
    où Jc = EpCost_moyen - cost_limit.
    
    Référence: Zhang et al. (2022), "Penalized Proximal Policy Optimization for Safe RL"

    [P3O] Paramètres additionnels par rapport à PPO:
        kappa: Coefficient de pénalité pour la violation de contrainte de coût (défaut: 20.0)
        cost_limit: Limite de coût cumulatif par épisode autorisée (défaut: 25.0)
        cost_vf_coef: Coefficient pour la loss du critique de coût (défaut: 0.5)
    """

    def __init__(
        self,
        policy: str | type,
        env: GymEnv | str,
        # [P3O] Paramètres spécifiques à P3O
        kappa: float = 20.0,
        cost_limit: float = 25.0,
        cost_vf_coef: float = 0.5,
        # Paramètres PPO hérités
        learning_rate: float | Schedule = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float | Schedule = 0.2,
        clip_range_vf: None | float | Schedule = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        rollout_buffer_class: type[RolloutBuffer] | None = None,
        rollout_buffer_kwargs: dict[str, Any] | None = None,
        target_kl: float | None = None,
        stats_window_size: int = 100,
        tensorboard_log: str | None = None,
        policy_kwargs: dict[str, Any] | None = None,
        verbose: int = 0,
        seed: int | None = None,
        device: th.device | str = "auto",
        _init_setup_model: bool = True,
    ):
        """
        Initialisation de P3O avec les paramètres PPO standard et les paramètres additionnels pour la pénalité de coût.

        Args:
            policy: Architecture de la politique (ex: "MlpPolicy")
            env: Environnement d'entraînement
            kappa: Coefficient de pénalité pour les violations de coût
            cost_limit: Limite de coût cumulatif par épisode
            cost_vf_coef: Coefficient pour la loss du critique de coût
            ... (autres paramètres PPO hérités)
        """
        # [P3O] Stocker les paramètres P3O avant l'init parent
        self.kappa = kappa
        self.cost_limit = cost_limit
        self.cost_vf_coef = cost_vf_coef

        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            clip_range_vf=clip_range_vf,
            normalize_advantage=normalize_advantage,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            # [P3O] Forcer l'utilisation du CostRolloutBuffer
            rollout_buffer_class=rollout_buffer_class or CostRolloutBuffer,
            rollout_buffer_kwargs=rollout_buffer_kwargs,
            target_kl=target_kl,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model,
        )

    def _setup_model(self) -> None:
        super()._setup_model()

        # [P3O] Construire le critique de coût (wrapper autour de la politique existante)
        # Réutilise les mêmes dimensions que le critique de récompense
        vf_arch = self.policy.net_arch
        if isinstance(vf_arch, dict):
            cost_arch = vf_arch.get("vf", [64, 64])
        elif isinstance(vf_arch, list):
            cost_arch = vf_arch
        else:
            cost_arch = [64, 64]

        self.cost_critic = CostCritic(
            features_dim=self.policy.features_dim,
            net_arch=cost_arch,
            activation_fn=self.policy.activation_fn,
        ).to(self.device)

        # Ici l'optimiseur ne mettra pas à jour les paramètres du feature_extractor car il ne contient aucun paramètres à entraîner (one-hot vector et flatten)
        self.cost_critic_optimizer = self.policy.optimizer_class(
            self.cost_critic.parameters(),
            lr=self.learning_rate,
            **self.policy.optimizer_kwargs,
        )

        # [P3O] Suivi du coût moyen par épisode pour le calcul de Jc
        self._ep_cost_buffer: deque[float] = deque(maxlen=self._stats_window_size)
        self._current_ep_costs = np.zeros(self.n_envs)

    def _predict_cost_values(self, obs: th.Tensor) -> th.Tensor:
        """[P3O] Prédit V_c(s) en réutilisant l'extracteur de features de la politique."""
        """Dans notre cas les features sont simplement un one-hot vector représentant la case sur la grille, et il n'y a pas de réseau extracteur de features."""
        features = self.policy.extract_features(obs, self.policy.vf_features_extractor)
        return self.cost_critic(features)

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: CostRolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        Collecte des rollouts en étendant la méthode PPO pour aussi collecter les coûts.
        
        [P3O] Différences par rapport à PPO:
        - Extraction du coût depuis info['cost'] à chaque pas
        - Prédiction de V_c(s) avec le critique de coût
        - Bootstrapping des coûts pour les épisodes tronqués
        - Calcul des avantages de coût via GAE
        - Suivi du coût cumulatif par épisode
        """
        assert self._last_obs is not None
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()

        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor)
                # [P3O] Prédire la valeur de coût
                cost_values = self._predict_cost_values(obs_tensor)

            actions = actions.cpu().numpy()
            clipped_actions = actions

            if isinstance(self.action_space, spaces.Box):
                if self.policy.squash_output:
                    clipped_actions = self.policy.unscale_action(clipped_actions)
                else:
                    clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            self.num_timesteps += env.num_envs

            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                actions = actions.reshape(-1, 1)

            # [P3O] Extraire les coûts depuis les infos
            costs = np.array([info.get("cost", 0.0) for info in infos], dtype=np.float32)

            # [P3O] Suivi du coût cumulatif par épisode
            self._current_ep_costs += costs
            for idx, done in enumerate(dones):
                if done:
                    self._ep_cost_buffer.append(self._current_ep_costs[idx])
                    self._current_ep_costs[idx] = 0.0

            # Bootstrapping pour les épisodes tronqués (récompenses ET coûts)
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]
                        # [P3O] Bootstrapping du coût
                        terminal_cost_value = self._predict_cost_values(terminal_obs)[0]
                    rewards[idx] += self.gamma * terminal_value
                    costs[idx] += self.gamma * terminal_cost_value.item() #[P3O]

            # [P3O] Ajout au buffer avec les coûts
            rollout_buffer.add(
                self._last_obs,
                actions,
                rewards,
                self._last_episode_starts,
                values,
                log_probs,
                cost=costs,
                cost_value=cost_values,
            )
            self._last_obs = new_obs
            self._last_episode_starts = dones

        with th.no_grad():
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))
            # [P3O] Valeur de coût pour le dernier pas
            cost_values = self._predict_cost_values(obs_as_tensor(new_obs, self.device))

        # [P3O] Calcul des retours et avantages (récompenses + coûts)
        rollout_buffer.compute_returns_and_advantage(
            last_values=values, dones=dones, last_cost_values=cost_values
        )

        callback.update_locals(locals())
        callback.on_rollout_end()

        return True

    def train(self) -> None:
        """
        Mise à jour de la politique avec la pénalité de coût P3O.
        
        [P3O] Différences par rapport à PPO:
        - Calcul de la loss de coût: L_cost = kappa * ReLU(surr_cost_adv + Jc)
        - Mise à jour du critique de coût (loss MSE séparée)
        - La loss totale de l'acteur est: L_ppo + L_cost
        """
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        # [P3O] Mettre à jour le learning rate du critique de coût
        for param_group in self.cost_critic_optimizer.param_groups:
            param_group["lr"] = self.policy.optimizer.param_groups[0]["lr"]

        clip_range = self.clip_range(self._current_progress_remaining)
        # Optional: clip range for the value function (non utilisé et non implémenté pour la fonction de coût)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        # [P3O] Calculer Jc = coût moyen par épisode - limite de coût
        if len(self._ep_cost_buffer) > 0:
            mean_ep_cost = np.mean(list(self._ep_cost_buffer))
        else:
            mean_ep_cost = 0.0
        jc = mean_ep_cost - self.cost_limit

        entropy_losses = []
        pg_losses, value_losses = [], []
        cost_losses = []          # [P3O]
        cost_value_losses = []    # [P3O]
        clip_fractions = []

        continue_training = True

        for epoch in range(self.n_epochs):
            approx_kl_divs = []

            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()

                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # === Loss PPO standard (récompense) ===
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    values_pred = values
                else: #(clipping pour la fonction de valeur non utilisé et non implémenté pour la fonction de coût)
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )

                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                if entropy is None:
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)
                entropy_losses.append(entropy_loss.item())

                # === [P3O] Loss de coût (pour la politique apprise) ===
                cost_adv = rollout_data.cost_advantages
                # [P3O] Normalisation des avantages de coût (même approche que pour les récompenses)
                if self.normalize_advantage and len(cost_adv) > 1:
                    cost_adv = (cost_adv - cost_adv.mean()) / (cost_adv.std() + 1e-8)
                surr_cost_adv = (ratio * cost_adv).mean()
                cost_loss = self.kappa * F.relu(surr_cost_adv + jc)
                cost_losses.append(cost_loss.item())

                # === [P3O] Loss du critique de coût ===
                features = self.policy.extract_features(
                    rollout_data.observations, self.policy.vf_features_extractor
                )
                cost_values_pred = self.cost_critic(features).flatten()
                cost_value_loss = F.mse_loss(rollout_data.cost_returns, cost_values_pred)
                cost_value_losses.append(cost_value_loss.item())

                # === Loss totale: PPO + pénalité de coût P3O ===
                # Critique updaté en même temps que l'acteur avec un gradient pondéré (vf_coef) et un optimizer partagé
                loss = (policy_loss # Politique
                        + self.ent_coef * entropy_loss # Politique
                        + cost_loss # [P3O] Politique
                        + self.vf_coef * value_loss) # Critique de récompense

                # Divergence KL approximative pour arrêt anticipé
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Arrêt anticipé à l'époque {epoch} (KL: {approx_kl_div:.2f})")
                    break

                # Mise à jour de la politique et du critique de récompense
                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

                # [P3O] Mise à jour séparée du critique de coût
                self.cost_critic_optimizer.zero_grad()
                cost_value_loss.backward()
                th.nn.utils.clip_grad_norm_(self.cost_critic.parameters(), self.max_grad_norm)
                self.cost_critic_optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten()
        )

        # Logs PPO standard
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

        # [P3O] Logs spécifiques
        self.logger.record("train/cost_loss", np.mean(cost_losses))
        self.logger.record("train/cost_value_loss", np.mean(cost_value_losses))
        self.logger.record("train/mean_ep_cost", mean_ep_cost)
        self.logger.record("train/jc", jc)


if __name__ == "__main__":
    import torch
    from environments import create_env

    print("Test rapide de P3O sur FrozenLake 8x8...")

    env = create_env("8x8_default", reward_type="CMDP", max_episode_steps=60,
                      success_rate=0.9, seed=42)

    policy_kwargs = dict(
        activation_fn=torch.nn.ReLU,
        net_arch=dict(pi=[16, 16], vf=[16, 16]),
    )

    model = P3O(
        "MlpPolicy", env, verbose=1,
        kappa=20.0, cost_limit=25.0,
        n_steps=256, batch_size=64, n_epochs=4,
        seed=42, device="cpu",
        policy_kwargs=policy_kwargs,
    )

    model.learn(total_timesteps=2048)

    env.close()
    print("Test terminé.")
