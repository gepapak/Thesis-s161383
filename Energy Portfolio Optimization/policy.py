from __future__ import annotations

import collections
from functools import partial
from typing import Any, Optional, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch import nn
from torch.distributions import Beta

from stable_baselines3.common.distributions import Distribution, sum_independent_dims
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import FlattenExtractor
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule


class RuleBasedPolicy:
    """
    Minimal non-learning policy wrapper for deterministic controllers.
    """

    def __init__(self, agent_name: str, observation_space: Any, action_space: Any, env: Any):
        self.agent_name = agent_name
        self.mode = "RULE"
        self.observation_space = observation_space
        self.action_space = action_space
        self.env = env
        self.device = "cpu"
        self.callback = None
        self.num_timesteps = 0

    def _resolve_env(self) -> Any:
        env = self.env
        for attr in ("base_env", "env"):
            try:
                next_env = getattr(env, attr, None)
                if next_env is not None:
                    env = next_env
            except Exception:
                pass
        return env

    def predict(self, obs, deterministic: bool = True):
        del deterministic
        env = self._resolve_env()
        action = None
        if env is not None and hasattr(env, "get_rule_based_agent_action"):
            action = env.get_rule_based_agent_action(self.agent_name, obs)
        if action is None:
            shape = getattr(self.action_space, "shape", (1,))
            action = np.zeros(shape, dtype=np.float32)
        action = np.asarray(action, dtype=np.float32).reshape(self.action_space.shape)
        return action, None


class BetaDistribution(Distribution):
    """
    Beta distribution on [-1, 1] obtained by affine-transforming a Beta(0, 1).
    """

    def __init__(self, action_dim: int, epsilon: float = 1e-6):
        super().__init__()
        self.action_dim = int(action_dim)
        self.epsilon = float(epsilon)
        self.alpha = None
        self.beta = None

    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        return nn.Linear(latent_dim, 2 * self.action_dim)

    def proba_distribution(self, action_logits: th.Tensor) -> "BetaDistribution":
        alpha_raw, beta_raw = th.chunk(action_logits, 2, dim=1)
        alpha = th.nn.functional.softplus(alpha_raw) + self.epsilon
        beta = th.nn.functional.softplus(beta_raw) + self.epsilon
        self.alpha = alpha
        self.beta = beta
        self.distribution = Beta(alpha, beta)
        return self

    def _to_unit_interval(self, actions: th.Tensor) -> th.Tensor:
        scaled = 0.5 * (actions + 1.0)
        return th.clamp(scaled, self.epsilon, 1.0 - self.epsilon)

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        unit_actions = self._to_unit_interval(actions)
        log_prob = self.distribution.log_prob(unit_actions) - np.log(2.0)
        return sum_independent_dims(log_prob)

    def entropy(self) -> Optional[th.Tensor]:
        return sum_independent_dims(self.distribution.entropy() + np.log(2.0))

    def sample(self) -> th.Tensor:
        unit = self.distribution.rsample()
        return 2.0 * unit - 1.0

    def mode(self) -> th.Tensor:
        mean = self.distribution.mean
        return 2.0 * mean - 1.0

    def actions_from_params(self, action_logits: th.Tensor, deterministic: bool = False) -> th.Tensor:
        self.proba_distribution(action_logits)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(self, action_logits: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        actions = self.actions_from_params(action_logits)
        log_prob = self.log_prob(actions)
        return actions, log_prob


class BetaActorCriticPolicy(ActorCriticPolicy):
    """
    SB3-compatible actor-critic policy that uses a Beta action distribution for
    bounded continuous actions instead of a Gaussian.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[list[int], dict[str, list[int]]]] = None,
        activation_fn: type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: type[nn.Module] = FlattenExtractor,
        features_extractor_kwargs: Optional[dict[str, Any]] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[dict[str, Any]] = None,
        beta_epsilon: float = 1e-6,
    ):
        self.beta_epsilon = float(beta_epsilon)
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            ortho_init=ortho_init,
            use_sde=use_sde,
            log_std_init=log_std_init,
            full_std=full_std,
            use_expln=use_expln,
            squash_output=squash_output,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            share_features_extractor=share_features_extractor,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
        )

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()
        default_none_kwargs = self.dist_kwargs or collections.defaultdict(lambda: None)  # type: ignore[arg-type, return-value]
        data.update(
            dict(
                beta_epsilon=self.beta_epsilon,
                net_arch=self.net_arch,
                activation_fn=self.activation_fn,
                use_sde=self.use_sde,
                log_std_init=self.log_std_init,
                squash_output=default_none_kwargs["squash_output"],
                full_std=default_none_kwargs["full_std"],
                use_expln=default_none_kwargs["use_expln"],
                lr_schedule=self._dummy_schedule,
                ortho_init=self.ortho_init,
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
            )
        )
        return data

    def _build(self, lr_schedule: Schedule) -> None:
        self._build_mlp_extractor()
        latent_dim_pi = self.mlp_extractor.latent_dim_pi
        self.action_dist = BetaDistribution(get_action_dim(self.action_space), epsilon=self.beta_epsilon)
        self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)

        if self.ortho_init:
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,
            }
            if not self.share_features_extractor:
                del module_gains[self.features_extractor]
                module_gains[self.pi_features_extractor] = np.sqrt(2)
                module_gains[self.vf_features_extractor] = np.sqrt(2)
            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)  # type: ignore[call-arg]

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor) -> Distribution:
        action_logits = self.action_net(latent_pi)
        return self.action_dist.proba_distribution(action_logits)

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))  # type: ignore[misc]
        return actions, values, log_prob

    def _predict(self, observation: PyTorchObs, deterministic: bool = False) -> th.Tensor:
        return self.get_distribution(observation).get_actions(deterministic=deterministic)


__all__ = ["RuleBasedPolicy", "BetaDistribution", "BetaActorCriticPolicy"]
