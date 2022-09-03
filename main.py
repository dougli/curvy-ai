import asyncio
import json
import time

import numpy as np
import torch
import torch.backends.mps
import torchvision.transforms as transforms

import logger
from ppo_torch import Agent
from screen import INPUT_SHAPE, Action, Game

REWARD_HISTORY_FILE = "out/reward_history.json"

# Hyperparameters
horizon = 128
lr = 0.0003
n_epochs = 3
minibatch_size = 32
gamma = 0.99
gae_lambda = 0.95
policy_clip = 0.1
vf_coeff = 1
entropy_coeff = 0.01


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


transform = transforms.ToTensor()


async def main():
    agent = Agent(
        device=init_device(),
        n_actions=len(Action),
        input_shape=INPUT_SHAPE,
        gamma=gamma,
        gae_lambda=gae_lambda,
        policy_clip=policy_clip,
        minibatch_size=minibatch_size,
        n_epochs=n_epochs,
        alpha=lr,
        vf_coeff=vf_coeff,
        entropy_coeff=entropy_coeff,
    )

    game = Game("lidouglas@gmail.com", "mzk-drw-krd3EVP5axn", show_screen=True)
    await game.launch()
    await game.login()
    await game.create_match()
    await game.start_match()

    # best_score = 0
    reward_history = []

    n_steps = 0
    avg_score = 0
    learn_iters = 0

    for episode in range(10000):
        await game.wait_for_alive()

        total_reward = 0
        done = False
        while not done:
            state = game.play_area
            action, prob, value = agent.choose_action(state)
            reward, done = await game.step(Action(action))

            n_steps += 1
            total_reward += reward
            agent.remember(state, action, prob, value, reward, done)

            if n_steps % horizon == 0:
                start_time = time.time()
                agent.learn()
                logger.success(f"Learned in {time.time() - start_time:.2f}s")
                learn_iters += 1

        reward_history.append(total_reward)
        avg_score = np.mean(reward_history[-100:])

        logger.info(
            f"Episode {episode}, reward: {round(total_reward, 3)}, Avg Score: {round(avg_score, 3)}, Time Steps: {n_steps}, Learn Iters: {learn_iters}"
        )

        # if avg_score > best_score:
        #     best_score = avg_score
        agent.save_models()

        with open(REWARD_HISTORY_FILE, "w") as f:
            json.dump(reward_history, f)


asyncio.run(main())
