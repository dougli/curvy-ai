from audioop import avg

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import constants
import utils
from agent import ImpalaCNN
from screen import INPUT_SHAPE, Action


def conv_0_filters():
    model = ImpalaCNN((4, 84, 84), len(Action))
    model.load_checkpoint()

    weights: np.ndarray = model.feat_convs[0][0].weight.detach().numpy()  # type: ignore
    weights = weights.squeeze()
    figure, axarr = plt.subplots(4 * 2, 4)

    for row in range(4):
        for frame in range(2):
            for col in range(4):
                axarr[row * 2 + frame][col].imshow(weights[row * 4 + col][frame])
                axarr[row * 2 + frame][col].axis("off")
    figure.show()

    return weights


def time_alive(avg_n=500):
    reward_history = utils.load_json(constants.REWARD_HISTORY_FILE, default=[])

    x = np.arange(len(reward_history))
    series = pd.Series([h["time"] for h in reward_history])

    series.plot(label="Time alive")
    series.rolling(avg_n).mean().plot(label=f"Running avg ({avg_n})")
    plt.title("Time alive")
    plt.show()


def reward(avg_n=500):
    reward_history = utils.load_json(constants.REWARD_HISTORY_FILE, default=[])

    x = np.arange(len(reward_history))
    series = pd.Series([h["reward"] for h in reward_history])

    mean = series.rolling(avg_n).mean()

    std = series.rolling(avg_n).std()

    mean.plot()
    plt.title(f"Average reward over {avg_n} games")
    plt.fill_between(x, mean - std, mean + std, alpha=0.5)  # type: ignore
    plt.show()


def training_loss(window=10):
    training_history = utils.load_json(constants.TRAIN_LOSS_HISTORY_FILE, default=[])
    x = np.arange(len(training_history))
    actor_loss = pd.Series([h["actor_loss"] for h in training_history])
    critic_loss = pd.Series([h["critic_loss"] for h in training_history])
    entropy = pd.Series([h["entropy_loss"] for h in training_history]) * 0.01
    actor_loss.plot(label="Actor Loss")
    actor_loss.rolling(window).mean().plot(label=f"Actor loss avg")
    critic_loss.plot(label="Critic Loss")
    critic_loss.rolling(window).mean().plot(label=f"Critic loss avg")
    entropy.plot(label="Entropy")
    entropy.rolling(window).mean().plot(label=f"Entropy avg")
    plt.title("Training Loss")
    plt.legend()
    plt.show()


def total_time_played():
    reward_history = utils.load_json(constants.REWARD_HISTORY_FILE, default=[])
    total_time = 0
    for entry in reward_history:
        total_time += entry["time"]

    return total_time


def simulate_random_old_agents(n=100, matches_each=15, lr=0.05):
    logits = []
    for i in range(n):
        logits.append(0 if i == 0 else max(logits))

        # Randomly play some games
        for j in range(matches_each):
            # Select an old agent using softmax probs
            probs = np.exp(logits) / np.sum(np.exp(logits))
            idx = np.random.choice(i + 1, p=probs)
            if np.random.randint(0, 2) == 1:
                logits[idx] = logits[idx] - lr / (len(logits) * probs[idx])

    probs = np.exp(logits) / np.sum(np.exp(logits))
    return probs


def old_agents_probability(probs):
    x = np.arange(len(probs))
    plt.bar(x, [*probs.values()])
    plt.show()
