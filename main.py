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

env = gym.make("CartPole-v1", new_step_api=True, render_mode="human")

# Hyperparameters
lr = 3e-4
minibatch_size = 32
epsilon = 0.1
discount = 0.99


def init_device():
    cuda = torch.cuda.is_available()  # Nvidia GPU
    mps = torch.backends.mps.is_available()  # M1 processors
    if cuda:
        print("Using CUDA")
        device = torch.device("cuda")
    elif mps:
        print("Using MPS")
        device = torch.device("mps")
    else:
        print("Using CPU")
        device = torch.device("cpu")
    return device


def run(device):
    for iteration in range(10000):
        state = env.reset()
        next_state = None
        done = False
        total_reward = 0
        while not done:
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            dist, value = agent.act(state)
            action = dist.sample()
            next_state, reward, term, trunc, info = env.step(action.item())  # type: ignore
            env.render()
            done = bool(term or trunc)
            if done:
                reward = -1
            agent.remember(state, action, dist.log_prob(action), value, reward, done)
            total_reward += reward
            state = next_state
        next_state = torch.FloatTensor(next_state).unsqueeze(0).to(device)
        agent.learn(next_state)
        print(f"Iteration {iteration} reward: {total_reward}")


device = init_device()
model = CurvyNet(4, 24, 2)
model.to(device)
agent = Agent(model, device)
run(device)
