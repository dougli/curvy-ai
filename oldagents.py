import os
import random

import numpy as np

import utils

OLD_AGENT_LEARN_RATE = 0.15


def old_agent_probs(folder: str):
    old_agents_reward_history = utils.load_json(
        f"{folder}/models/old/reward_history.json", default=[]
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
            logits[agent] -= OLD_AGENT_LEARN_RATE / (len(logits) * prob)

    # If there is an agent that hasn't been played yet, initialize it to the max chance
    available_old_agents = list_old_agents(folder)
    for agent in available_old_agents:
        if agent not in logits:
            if not logits:
                logits[agent] = 0
            else:
                logits[agent] = max([*logits.values()])
    total_prob = np.sum(np.exp([*logits.values()]))
    probs = {agent: np.exp(logits[agent]) / total_prob for agent in logits}
    return probs


def list_old_agents(folder: str) -> list[str]:
    directory = os.path.join(folder, "models", "old")
    if not os.path.exists(directory):
        return []
    backups = os.listdir(directory)
    backups = [backup for backup in backups if backup.endswith(".zip")]
    return backups


def select_old_agent(folder: str):
    probs = old_agent_probs(folder)
    if not probs:
        return None
    agent = random.choices(list(probs.keys()), list(probs.values()))[0]
    return agent


def win_loss_ratio(folder: str):
    old_agents_reward_history = utils.load_json(
        f"{folder}/models/old/reward_history.json", default=[]
    )
    wins = 0
    losses = 0
    for entry in old_agents_reward_history:
        if entry["won"]:
            wins += 1
        else:
            losses += 1
    return {"losses": wins, "wins": losses}


def win_loss_ratio_per_agent(folder: str):
    old_agents_reward_history = utils.load_json(
        f"{folder}/models/old/reward_history.json", default=[]
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
