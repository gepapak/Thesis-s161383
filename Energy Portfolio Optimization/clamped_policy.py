"""
Standalone clamped PPO policy.

This module keeps the anti-collapse PPO policy logic decoupled from optional
feature-extractor implementations, so core runs remain stable and portable.
"""

from __future__ import annotations

import torch
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import (
    DiagGaussianDistribution,
    StateDependentNoiseDistribution,
)


class ClampedActorCriticPolicy(ActorCriticPolicy):
    """
    PPO policy that stabilizes continuous actions by clamping:
    - mean logits before tanh squashing
    - learned log-std range
    - optional gSDE latent input
    """

    def __init__(self, *args, **kwargs):
        log_std_min_raw = kwargs.pop("log_std_min", -2.0)
        log_std_max_raw = kwargs.pop("log_std_max", -0.5)
        mean_clip_raw = kwargs.pop("mean_clip", 0.85)
        sde_latent_clip_raw = kwargs.pop("sde_latent_clip", 0.5)

        self._log_std_min = float(log_std_min_raw)
        self._log_std_max = float(log_std_max_raw)
        if self._log_std_min > self._log_std_max:
            raise ValueError(
                f"log_std_min ({self._log_std_min}) cannot exceed log_std_max ({self._log_std_max})."
            )

        if mean_clip_raw is None:
            self._mean_clip = None
        else:
            self._mean_clip = float(mean_clip_raw)
            if self._mean_clip <= 0.0:
                self._mean_clip = None

        if sde_latent_clip_raw is None:
            self._sde_latent_clip = None
        else:
            self._sde_latent_clip = float(sde_latent_clip_raw)
            if self._sde_latent_clip <= 0.0:
                self._sde_latent_clip = None
        super().__init__(*args, **kwargs)

    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor):
        mean_actions = self.action_net(latent_pi)
        if self._mean_clip is not None and self._mean_clip > 0:
            mean_actions = torch.clamp(mean_actions, -self._mean_clip, self._mean_clip)

        if isinstance(self.action_dist, DiagGaussianDistribution):
            log_std = self.log_std
            if self._log_std_min is not None or self._log_std_max is not None:
                log_std = torch.clamp(
                    log_std,
                    min=self._log_std_min if self._log_std_min is not None else None,
                    max=self._log_std_max if self._log_std_max is not None else None,
                )
            return self.action_dist.proba_distribution(mean_actions, log_std)

        if isinstance(self.action_dist, StateDependentNoiseDistribution):
            log_std = self.log_std
            if self._log_std_min is not None or self._log_std_max is not None:
                log_std = torch.clamp(
                    log_std,
                    min=self._log_std_min if self._log_std_min is not None else None,
                    max=self._log_std_max if self._log_std_max is not None else None,
                )
            latent_sde = latent_pi
            if self._sde_latent_clip is not None:
                latent_sde = torch.clamp(
                    latent_sde, -self._sde_latent_clip, self._sde_latent_clip
                )
            return self.action_dist.proba_distribution(mean_actions, log_std, latent_sde)

        return super()._get_action_dist_from_latent(latent_pi)
