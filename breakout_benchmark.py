import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Parallel environments
env = make_vec_env("BreakoutNoFrameskip-v4", n_envs=8)

model = PPO(
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
)
model.learn(total_timesteps=int(1e7))
model.save("ppo_breakout")

del model  # remove to demonstrate saving and loading

model = PPO.load("ppo_breakout")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
