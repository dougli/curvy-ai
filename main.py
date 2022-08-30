import time

import gym
import numpy as np
import torch
import torch.backends.mps

from agent.agent import Agent
from agent.net import CurvyNet
from screen import screen

# frames = 0
# start_time = time.time()
# while True:
#     screen.frame()
#     image = screen.game_area()
#     score = screen.game_state()

#     now = time.time()
#     frames += 1
#     fps = round(frames / (now - start_time), 2)
#     print(f"FPS: {fps}")

env = gym.make("CartPole-v1", new_step_api=True)


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


T = 20


def run(device):
    iteration = 0
    state = env.reset()
    next_state = None
    total_reward = 0
    steps = 0
    learned_steps = 0

    while iteration < 10000:
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        dist, value = agent.act(state)
        action = dist.sample()
        next_state, reward, term, trunc, info = env.step(action.item())  # type: ignore
        done = bool(term or trunc)
        agent.remember(state, action, dist.log_prob(action), value, reward, done)
        total_reward += reward
        state = env.reset() if done else next_state

        steps += 1

        if done:
            print(
                f"Iteration: {iteration}, Reward: {total_reward}, Learning steps: {learned_steps}"
            )
            iteration += 1
            total_reward = 0

        if steps % T == 0:
            next_state = torch.FloatTensor(next_state).unsqueeze(0).to(device)
            agent.learn(next_state)
            steps = 0
            learned_steps += 1


device = init_device()
model = CurvyNet(4, 24, 2)
model.to(device)
agent = Agent(model, device)
run(device)
