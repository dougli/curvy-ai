from typing import Generator, Optional, Union

import numpy as np
import torch as th
from gym import spaces
from stable_baselines3.common.buffers import BaseBuffer
from stable_baselines3.common.type_aliases import RolloutBufferSamples


class AsyncRolloutBuffer:
    def __init__(
        self,
        max_buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):
        self.max_buffer_size = max_buffer_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = device
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.n_envs = n_envs

        self.buffers = [
            SingleAsyncBuffer(
                max_buffer_size,
                observation_space,
                action_space,
                device,
                gae_lambda,
                gamma,
            )
            for _ in range(n_envs)
        ]

    def reset(self) -> None:
        for buffer in self.buffers:
            buffer.reset()

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: th.Tensor,
        log_prob: th.Tensor,
        env_num: int,
    ):
        self.buffers[env_num].add(obs, action, reward, episode_start, value, log_prob)

    def get(
        self, batch_size: Optional[int] = None
    ) -> Generator[RolloutBufferSamples, None, None]:
        rollout_data: list[RolloutBufferSamples] = []
        for buffer in self.buffers:
            rollout_data.append(*buffer.get())

        observations = th.cat([r.observations for r in rollout_data], dim=0)
        actions = th.cat([r.actions for r in rollout_data], dim=0)
        old_values = th.cat([r.old_values for r in rollout_data], dim=0)
        old_log_prob = th.cat([r.old_log_prob for r in rollout_data], dim=0)
        advantages = th.cat([r.advantages for r in rollout_data], dim=0)
        returns = th.cat([r.returns for r in rollout_data], dim=0)

        self.values = old_values.numpy()
        self.returns = returns.numpy()

        indices = np.random.permutation(returns.shape[0])

        if batch_size is None:
            batch_size = returns.shape[0]

        start_idx = 0
        while start_idx + batch_size <= returns.shape[0]:
            yield RolloutBufferSamples(
                observations=observations[indices[start_idx : start_idx + batch_size]],
                actions=actions[indices[start_idx : start_idx + batch_size]],
                old_values=old_values[indices[start_idx : start_idx + batch_size]],
                old_log_prob=old_log_prob[indices[start_idx : start_idx + batch_size]],
                advantages=advantages[indices[start_idx : start_idx + batch_size]],
                returns=returns[indices[start_idx : start_idx + batch_size]],
            )
            start_idx += batch_size

    def compute_returns_and_advantage(self) -> None:
        for buffer in self.buffers:
            buffer.compute_returns_and_advantage()


class SingleAsyncBuffer(BaseBuffer):
    def __init__(
        self,
        max_buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        gae_lambda: float = 1,
        gamma: float = 0.99,
    ):
        super().__init__(
            max_buffer_size, observation_space, action_space, device, n_envs=1
        )
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.reset()

    def reset(self) -> None:
        self.observations = np.zeros(
            (self.buffer_size,) + self.obs_shape, dtype=np.float32  # type: ignore
        )
        self.actions = np.zeros((self.buffer_size, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size,), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size,), dtype=np.float32)
        self.episode_starts = np.zeros((self.buffer_size,), dtype=np.float32)
        self.values = np.zeros((self.buffer_size,), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size,), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size,), dtype=np.float32)
        self.pos = 0
        super().reset()

    def compute_returns_and_advantage(self) -> None:
        """
        Post-processing step: compute the lambda-return (TD(lambda) estimate)
        and GAE(lambda) advantage.

        Uses Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)
        to compute the advantage. To obtain Monte-Carlo advantage estimate (A(s) = R - V(S))
        where R is the sum of discounted reward with value bootstrap
        (because we don't always have full episode), set ``gae_lambda=1.0`` during initialization.

        The TD(lambda) estimator has also two special cases:
        - TD(1) is Monte-Carlo estimate (sum of discounted rewards)
        - TD(0) is one-step estimate with bootstrapping (r_t + gamma * v(s_{t+1}))

        For more information, see discussion in https://github.com/DLR-RM/stable-baselines3/pull/375.

        :param last_values: state value estimation for the last step (one for each env)
        :param dones: if the last step was a terminal step (one bool for each env).
        """
        last_gae_lam = 0

        # Discard the last element of the trajectory because we don't have the next value
        for step in reversed(range(self.pos - 1)):
            assert step < self.pos - 1
            next_non_terminal = 1.0 - self.episode_starts[step + 1]
            next_values = self.values[step + 1]
            delta = (
                self.rewards[step]
                + self.gamma * next_values * next_non_terminal
                - self.values[step]
            )
            last_gae_lam = (
                delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            )
            self.advantages[step] = last_gae_lam
        # TD(lambda) estimator, see Github PR #375 or "Telescoping in TD(lambda)"
        # in David Silver Lecture 4: https://www.youtube.com/watch?v=PnHCvfgC_ZA
        self.returns = self.advantages + self.values

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: th.Tensor,
        log_prob: th.Tensor,
    ) -> None:
        """
        :param obs: Observation
        :param action: Action
        :param reward:
        :param episode_start: Start of episode signal.
        :param value: estimated value of the current state
            following the current policy.
        :param log_prob: log probability of the action
            following the current policy.
        """
        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)

        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape(self.obs_shape)  # type: ignore

        # Same reshape, for actions
        action = action.reshape((1, self.action_dim))

        self.observations[self.pos] = np.array(obs).copy()
        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.episode_starts[self.pos] = np.array(episode_start).copy()
        self.values[self.pos] = value.clone().cpu().numpy().flatten()
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy()
        self.pos += 1

    def get(self) -> Generator[RolloutBufferSamples, None, None]:
        indices = np.random.permutation(self.pos - 1)
        # Return everything, don't create minibatches
        batch_size = self.pos - 1
        yield self._get_samples(indices[0:batch_size])

    def _get_samples(self, batch_inds: np.ndarray) -> RolloutBufferSamples:
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
        )
        return RolloutBufferSamples(*tuple(map(self.to_torch, data)))
