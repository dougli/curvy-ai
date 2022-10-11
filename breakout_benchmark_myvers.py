import torch as th
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage

from agent.stable_agent import StableAgent

SEED = 12345

env = make_vec_env(
    "BreakoutNoFrameskip-v4", n_envs=8, seed=SEED, wrapper_class=AtariWrapper
)

env = VecFrameStack(env, n_stack=4)
env = VecTransposeImage(env)
print("Environment dimensions: ", env.observation_space)


model = StableAgent(
    "CnnPolicy",
    env,
    verbose=1,
    n_steps=128,
    n_epochs=4,
    batch_size=256,
    learning_rate=(lambda rem: 2.5e-4 * rem),
    clip_range=0.1,
    vf_coef=0.5,
    ent_coef=0.01,
    seed=SEED,
)
model.setup_learn(total_timesteps=int(1e7))

obs = env.reset()
prev_dones = [False for _ in range(env.num_envs)]
while model.num_timesteps < model.total_timesteps:
    # Collect rollouts
    model.policy.set_training_mode(False)
    n_steps = 0
    model.rollout_buffer.reset()  # type: ignore
    while n_steps < model.n_steps:
        with th.no_grad():
            # Convert to pytorch tensor or to TensorDict
            obs_tensor = obs_as_tensor(obs, model.device)  # type: ignore
            actions, values, log_probs = model.policy(obs_tensor)
        actions = actions.cpu().numpy()

        new_obs, rewards, dones, infos = env.step(actions)

        model.num_timesteps += env.num_envs
        # Give access to local variables
        model.callback.update_locals(locals())
        if model.callback.on_step() is False:
            raise Exception("We should shop due to callback")

        model._update_info_buffer(infos)
        n_steps += 1
        actions = actions.reshape(-1, 1)
        model.rollout_buffer.add(
            obs,
            actions,
            rewards,
            prev_dones,
            values,
            log_probs,
        )
        obs = new_obs
        prev_dones = dones

    with th.no_grad():
        # Compute value for the last timestep
        values = model.policy.predict_values(obs_as_tensor(obs, model.device))

    model.rollout_buffer.compute_returns_and_advantage(
        last_values=values, dones=prev_dones
    )

    model.callback.on_rollout_end()

    # ...

    model.learn()

model.save("ppo_breakout")

del model  # remove to demonstrate saving and loading

model = StableAgent.load("ppo_breakout")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
