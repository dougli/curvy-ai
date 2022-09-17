import matplotlib.pyplot as plt
import numpy as np

import constants
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
    reward_history = utils.load_json(constants.REWARD_HISTORY_FILE, default=[])

    x = np.arange(len(reward_history))
    time_alive = [h["time"] for h in reward_history]

    plt.plot(x, time_alive, label="Time alive")
    plt.plot(x, running_avg(time_alive), label="Running Avg (100)")
    plt.title("Time alive")
    plt.show()


def reward():
    reward_history = utils.load_json(constants.REWARD_HISTORY_FILE, default=[])

    x = np.arange(len(reward_history))
    rewards = [h["reward"] for h in reward_history]

    plt.plot(x, rewards, label="Reward")
    plt.plot(x, running_avg(rewards), label="Running Avg (100)")
    plt.title("Reward")
    plt.show()


def training_loss():
    training_history = utils.load_json(constants.TRAIN_LOSS_HISTORY_FILE, default=[])
    x = np.arange(len(training_history))
    actor_loss = [h["actor_loss"] for h in training_history]
    critic_loss = [h["critic_loss"] for h in training_history]
    entropy_loss = [h["entropy_loss"] for h in training_history]
    plt.plot(x, actor_loss, label="Actor Loss")
    plt.plot(x, running_avg(actor_loss), label="Actor loss avg")
    plt.plot(x, critic_loss, label="Critic Loss")
    plt.plot(x, running_avg(critic_loss), label="Critic loss avg")
    plt.plot(x, entropy_loss, label="Entropy")
    plt.title("Training Loss")
    plt.legend()
    plt.show()


def total_time_played():
    reward_history = utils.load_json(constants.REWARD_HISTORY_FILE, default=[])
    total_time = 0
    for entry in reward_history:
        total_time += entry["time"]

    return total_time


def running_avg(values, n=100):
    running_avg = np.zeros(len(values))
    for i in range(len(values)):
        running_avg[i] = np.mean(values[max(0, i - n) : (i + 1)])
    return running_avg
