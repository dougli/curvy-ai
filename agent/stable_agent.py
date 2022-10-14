import sys
import time
from typing import Optional, TypeVar

import gym
import numpy as np
import torch as th
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback
from stable_baselines3.common.utils import obs_as_tensor, safe_mean
from stable_baselines3.common.vec_env.base_vec_env import VecEnv
from stable_baselines3.ppo.ppo import PPO

OnPolicyAlgorithmSelf = TypeVar("OnPolicyAlgorithmSelf", bound="OnPolicyAlgorithm")


class StableAgent(PPO):
    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.
        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        # assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        # self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor)
            actions = actions.cpu().numpy()

            new_obs, rewards, dones, infos = env.step(actions)

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, gym.spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            rollout_buffer.add(
                self._last_obs,
                actions,
                rewards,
                self._last_episode_starts,
                values,
                log_probs,
            )
            self._last_obs = new_obs
            self._last_episode_starts = dones

        with th.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.on_rollout_end()

        return True

    def setup_learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "OnPolicyAlgorithm",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ):
        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            eval_env,
            callback,
            eval_freq,
            n_eval_episodes,
            eval_log_path,
            reset_num_timesteps,
            tb_log_name,
        )
        self.total_timesteps = total_timesteps
        self.callback = callback

        self.callback.on_training_start(locals(), globals())

        return self

    def learn(
        self,
        log_interval: int = 1,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "OnPolicyAlgorithm",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ):
        iteration = 0

        # while self.num_timesteps < total_timesteps:

        # continue_training = self.collect_rollouts(
        #     self.env, callback, self.rollout_buffer, n_rollout_steps=self.n_steps
        # )

        # if continue_training is False:
        #     break

        iteration += 1
        self._update_current_progress_remaining(
            self.num_timesteps, self.total_timesteps
        )

        self.train()

        # Display training infos
        if log_interval is not None and iteration % log_interval == 0:
            time_elapsed = max(
                (time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon
            )
            fps = int(
                (self.num_timesteps - self._num_timesteps_at_start) / time_elapsed
            )
            self.logger.record("time/iterations", iteration, exclude="tensorboard")
            if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                self.logger.record(
                    "rollout/ep_rew_mean",
                    safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]),
                )
                self.logger.record(
                    "rollout/ep_len_mean",
                    safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]),
                )
            self.logger.record("time/fps", fps)
            self.logger.record(
                "time/time_elapsed", int(time_elapsed), exclude="tensorboard"
            )
            self.logger.record(
                "time/total_timesteps", self.num_timesteps, exclude="tensorboard"
            )
            self.logger.dump(step=self.num_timesteps)

        self.callback.on_training_end()

        return self
