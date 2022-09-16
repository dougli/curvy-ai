import os
import random

import matplotlib.pyplot as plt
import numpy as np

import constants
import logger
import utils
from agent import CurvyNet
from screen import INPUT_SHAPE, Action


def conv_0_filters():
    model = CurvyNet((2, *INPUT_SHAPE), len(Action))
    model.load_checkpoint()

    weights: np.ndarray = model.shared[0].weight.detach().numpy()  # type: ignore
    weights = weights.squeeze()
    figure, axarr = plt.subplots(4 * 2, 8)

    for row in range(4):
        for frame in range(2):
            for col in range(8):
                axarr[row * 2 + frame][col].imshow(weights[row * 8 + col][frame])
                axarr[row * 2 + frame][col].axis("off")
    figure.show()

    return weights


def time_alive():
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


def total_time_played():
    reward_history = utils.load_json(constants.REWARD_HISTORY_FILE)
    total_time = 0
    for entry in reward_history:
        total_time += entry["time"]

    return total_time
