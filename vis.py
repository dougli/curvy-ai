import json
import random

import matplotlib.pyplot as plt
import numpy as np

import constants
import utils
from agent import CurvyNet
from screen import INPUT_SHAPE, Action


def conv_0_filters():
    model = CurvyNet((1, *INPUT_SHAPE), len(Action))
    model.load_checkpoint()

    weights: np.ndarray = model.shared[0].weight.detach().numpy()  # type: ignore
    weights = weights.squeeze()
    figure, axarr = plt.subplots(4, 8)

    for row in range(4):
        for col in range(8):
            axarr[row][col].imshow(weights[row * 8 + col])
            axarr[row][col].axis("off")

    figure.show()

    return weights


def rewards():
    reward_history = utils.load_json(constants.REWARD_HISTORY_FILE)

    x = np.arange(len(reward_history))
    time_alive = [h["time"] for h in reward_history]
    running_avg = np.zeros(len(reward_history))
    for i in range(len(time_alive)):
        running_avg[i] = np.mean(time_alive[max(0, i - 100) : (i + 1)])

    plt.plot(x, time_alive, label="Time alive")
    plt.plot(x, running_avg, label="Running Avg (100)")
    plt.title("Time alive")
    plt.show()


def reconstruct_old_agent_quality():
    old_agents_reward_history: dict[str, list] = utils.load_json(
        constants.OLD_AGENTS_REWARD_HISTORY_FILE
    )

    n_old_agents = len(old_agents_reward_history)
    n_games = sum(len(games) for games in old_agents_reward_history.values())

    games_per_agent = n_games / n_old_agents
    games_so_far = 0

    games_arr = []

    quality = {}
    for i in range(n_games):
        if not games_arr or i >= games_so_far + games_per_agent:
            # pop the first key off the dict
            key = next(iter(old_agents_reward_history))
            games = [
                (key, v["score"] >= 50) for v in old_agents_reward_history.pop(key)
            ]
            games_arr.extend(games)
            random.shuffle(games_arr)

            games_so_far = i

        # tuple of (agent_name, win)
        agent_name, agent_won = games_arr.pop()
        if agent_name not in quality:
            quality[agent_name] = max(quality.values(), default=0)
        if not agent_won:
            # Quality update is q_i = q_i - 0.01 / (N * p_i), where p_i is the
            # probability of the agent getting selected.
            # N is the number of agents so far in the quality dict.
            # p_i is softmax of the quality values.
            probs = np.exp([*quality.values()]) / sum(np.exp([*quality.values()]))
            p_i = probs[list(quality.keys()).index(agent_name)]
            quality[agent_name] -= 0.01 / (len(quality) * p_i)

    print(quality)
