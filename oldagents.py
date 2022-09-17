import os
import random

import numpy as np

import constants
import utils


def old_agent_probs():
    old_agents_reward_history = utils.load_json(
        constants.OLD_AGENTS_REWARD_HISTORY_FILE, default=[]
    )

    logits: dict[str, float] = {}

    for entry in old_agents_reward_history:
        agent = entry["agent_name"]

        if agent not in logits:
            if not logits:
                logits[agent] = 0
            else:
                logits[agent] = max([*logits.values()])

        if not entry["won"]:
            prob = np.exp(logits[agent]) / np.sum(np.exp([*logits.values()]))
            logits[agent] -= 0.01 / (len(logits) * prob)

    # If there is an agent that hasn't been played yet, initialize it to the max chance
    available_old_agents = list_old_agents()
    for agent in available_old_agents:
        if agent not in logits:
            if not logits:
                logits[agent] = 0
            else:
                logits[agent] = max([*logits.values()])
    total_prob = np.sum(np.exp([*logits.values()]))
    probs = {agent: np.exp(logits[agent]) / total_prob for agent in logits}
    return probs


def list_old_agents() -> list[str]:
    directory = os.path.join(os.path.dirname(__file__), constants.BACKUP_DIR)
    if not os.path.exists(directory):
        return []
    backups = os.listdir(directory)
    if ".DS_Store" in backups:
        backups.remove(".DS_Store")
    return backups


def select_old_agent():
    probs = old_agent_probs()
    if not probs:
        return None
    agent = random.choices(list(probs.keys()), list(probs.values()))[0]
    return agent


def win_loss_ratio():
    old_agents_reward_history = utils.load_json(
        constants.OLD_AGENTS_REWARD_HISTORY_FILE, default=[]
    )
    wins = 0
    losses = 0
    for entry in old_agents_reward_history:
        if entry["won"]:
            wins += 1
        else:
            losses += 1
    return {"losses": wins, "wins": losses}


def win_loss_ratio_per_agent():
    old_agents_reward_history = utils.load_json(
        constants.OLD_AGENTS_REWARD_HISTORY_FILE, default=[]
    )
    agents = {}
    for entry in old_agents_reward_history:
        agent = entry["agent_name"]
        if agent not in agents:
            agents[agent] = {"losses": 0, "wins": 0}
        if entry["won"]:
            agents[agent]["wins"] += 1
        else:
            agents[agent]["losses"] += 1
    return agents
