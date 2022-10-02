import asyncio
import json
import os

import gym
import torch
import torch.backends.mps

import logger
import utils
from agent import Agent

from stable_baselines3.common.atari_wrappers import (  # isort:skip
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)

# Hyperparameters
n_agents = 8
horizon = n_agents * (128 + 1)  # Add 1 to account for last state in the trajectory
lr = 2.5e-4
n_epochs = 3
minibatch_size = 32
gamma = 0.99
gae_lambda = 0.95
policy_clip = 0.1
vf_coeff = 1
entropy_coeff = 0.01

SAVE_MODEL_EVERY_N_TRAINS = 30
REWARD_HISTORY_FILE = "out/reward_history.json"


def init_device():
    cuda = torch.cuda.is_available()  # Nvidia GPU
    mps = torch.backends.mps.is_available()  # M1 processors
    if cuda:
        print("Using CUDA")
        device = torch.device("cuda")
    # elif mps:
    #     print("Using MPS")
    #     device = torch.device("mps")
    else:
        print("Using CPU")
        device = torch.device("cpu")
    return device


class Trainer:
    def __init__(self):
        self.reward_history = utils.load_json(REWARD_HISTORY_FILE, [])

        self.n_steps = 0
        self.n_trains = 0

    async def run(self):
        self.device = init_device()

        envs = []
        for i in range(n_agents):
            env = gym.make("BreakoutNoFrameskip-v4")
            env = NoopResetEnv(env, noop_max=30)
            env = MaxAndSkipEnv(env, skip=4)
            env = EpisodicLifeEnv(env)
            if "FIRE" in env.unwrapped.get_action_meanings():
                env = FireResetEnv(env)
            env = ClipRewardEnv(env)
            env = gym.wrappers.ResizeObservation(env, (84, 84))
            env = gym.wrappers.GrayScaleObservation(env)
            env = gym.wrappers.FrameStack(env, 4)
            envs.append(env)

        logger.info(f"Input shape: {envs[0].observation_space.shape}")
        self.agent = Agent(
            device=self.device,
            n_actions=envs[0].action_space.n,
            input_shape=envs[0].observation_space.shape,
            gamma=gamma,
            gae_lambda=gae_lambda,
            policy_clip=policy_clip,
            minibatch_size=minibatch_size,
            n_agents=n_agents,
            n_epochs=n_epochs,
            alpha=lr,
            vf_coeff=vf_coeff,
            entropy_coeff=entropy_coeff,
        )
        if not self.agent.load_models():
            self.agent.save_models()

            pass

        timesteps = self.reward_history[-1]["timesteps"] if self.reward_history else 0
        obs = [env.reset() for env in envs]
        total_rewards = [0 for _ in range(n_agents)]
        while timesteps < 1e8:
            for i, env in enumerate(envs):
                state = (
                    torch.tensor([obs[i]], dtype=torch.float32).to(self.device) / 255.0
                )
                action, probs, value = self.agent.model.choose_action(state)
                next_obs, reward, done, info = env.step(action)
                total_rewards[i] += reward
                self.remember(
                    i, state.numpy().squeeze(), action, probs, value, reward, done
                )
                timesteps += 1
                if done:
                    self.log_reward(
                        {
                            "timesteps": timesteps,
                            "reward": total_rewards[i],
                            "info": info,
                        }
                    )
                    next_obs = env.reset()
                    total_rewards[i] = 0
                obs[i] = next_obs

    def remember(self, id, state, action, probs, vals, reward, done):
        self.agent.remember(id, state, action, probs, vals, reward, done)
        self.n_steps += 1
        if self.n_steps % horizon == 0:
            logger.warning("Training agent")
            self.agent.learn()
            self.agent.save_models()
            self.n_trains += 1
            if self.n_trains % SAVE_MODEL_EVERY_N_TRAINS == 0:
                self.agent.backup_models()

    def log_reward(self, data):
        self.reward_history.append(data)
        logger.warning(f"Reward: {data}")
        utils.save_json(REWARD_HISTORY_FILE, self.reward_history)


if __name__ == "__main__":
    trainer = Trainer()
    asyncio.run(trainer.run())
