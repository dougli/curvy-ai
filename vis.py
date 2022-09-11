import json

import matplotlib.pyplot as plt
import numpy as np

from agent import CurvyNet
from screen import INPUT_SHAPE, Action

REWARD_HISTORY_FILE = "out/reward_history.json"


def conv_0_filters():
    model = CurvyNet((1, *INPUT_SHAPE), len(Action))
    model.load_latest_checkpoint()

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
    with open(REWARD_HISTORY_FILE, "r") as f:
        reward_history = json.load(f)

    x = np.arange(len(reward_history))
    running_avg = np.zeros(len(reward_history))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(reward_history[max(0, i - 100) : (i + 1)])

    plt.plot(x, reward_history)
    plt.plot(x, running_avg)
    plt.title("Reward History")
    plt.show()
