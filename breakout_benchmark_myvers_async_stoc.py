import random

import torch as th
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage

from agent.stable_agent import StableAgent
from async_rollout_buffer import AsyncRolloutBuffer

SEED = 12345

n_envs = 8

envs = []
obs = []
prev_dones = [False for _ in range(n_envs)]
for i in range(n_envs):
    env = make_vec_env("BreakoutNoFrameskip-v4", seed=SEED, wrapper_class=AtariWrapper)
    env = VecFrameStack(env, n_stack=4)
    env = VecTransposeImage(env)
    envs.append(env)
    obs.append(env.reset())

print("Environment dimensions: ", envs[0].observation_space)


model = StableAgent(
    "CnnPolicy",
    envs[0],
    verbose=1,
    n_steps=129,
    n_epochs=4,
    batch_size=256,
    learning_rate=(lambda rem: 2.5e-4 * rem),
    clip_range=0.1,
    vf_coef=0.5,
    ent_coef=0.01,
    seed=SEED,
    tensorboard_log="./logs",
)
model.rollout_buffer = AsyncRolloutBuffer(
    model.n_steps * 2,
    envs[0].observation_space,
    envs[0].action_space,
    device=model.device,
    gae_lambda=model.gae_lambda,
    gamma=model.gamma,
    n_envs=8,
)
model.setup_learn(total_timesteps=int(1e7))


while model.num_timesteps < model.total_timesteps:
    # Collect rollouts
    model.policy.set_training_mode(False)
    n_steps = 0
    model.rollout_buffer.reset()  # type: ignore
    while n_steps < model.n_steps * n_envs:
        idx = random.randint(0, n_envs - 1)

        with th.no_grad():
            # Convert to pytorch tensor or to TensorDict
            obs_tensor = obs_as_tensor(obs[idx], model.device)  # type: ignore
            actions, values, log_probs = model.policy(obs_tensor)
        actions = actions.cpu().numpy()

        new_obs, rewards, dones, infos = envs[idx].step(actions)

        model.num_timesteps += 1
        # Give access to local variables
        model.callback.update_locals(locals())
        if model.callback.on_step() is False:
            raise Exception("We should shop due to callback")

        model._update_info_buffer(infos)
        n_steps += 1
        actions = actions.reshape(-1, 1)
        model.rollout_buffer.add(
            obs[idx],
            actions[0],
            rewards[0],
            prev_dones[idx],  # type: ignore
            values[0],
            log_probs[0],
            env_num=idx,
        )

        obs[idx] = new_obs
        prev_dones[idx] = dones[0]

    model.rollout_buffer.compute_returns_and_advantage()

    model.callback.on_rollout_end()

    model.learn()

    model.save("ppo_breakout", exclude=["callback"])
    prev_callback = model.callback
    del model
    model = StableAgent.load("ppo_breakout", env=envs[0])
    model.rollout_buffer = AsyncRolloutBuffer(
        model.n_steps * 2,
        envs[0].observation_space,
        envs[0].action_space,
        device=model.device,
        gae_lambda=model.gae_lambda,
        gamma=model.gamma,
        n_envs=8,
    )
    model.setup_learn(total_timesteps=int(1e7), reset_num_timesteps=False)
