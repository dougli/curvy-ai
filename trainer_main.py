import asyncio
import json
import os

import torch
import torch.backends.mps

import logger
from agent import Agent
from screen import INPUT_SHAPE, Account, Action
from worker import WorkerProcess

ACCOUNTS = [
    # Account(
    #     "jotocik870@nicoimg.com",
    #     "DZA*vev-kvg3etx-twz",
    #     "sarcatan's match",
    #     "vQgJ3zaP",
    #     True,
    #     ("eafdorf",),
    # ),
    # Account(
    #     "wheels.hallow_0g@icloud.com",
    #     "8.h*oVJokU69.GuHyLMf",
    #     "sarcatan's match",
    #     "vQgJ3zaP",
    #     False,
    # ),
    Account(
        "misstep.gills.0x@icloud.com",
        ".-.qqn@F@64eiwYk*Mao",
        "feedx9",
        "18xielPa",
        True,
        ("coiifan",),
    ),
    Account(
        "woyexo5962@iunicus.com", "RQZ1tgt3etg3wfk-cmu", "feedx9", "18xielPa", False
    ),
]


# Hyperparameters
horizon = 512 + len(ACCOUNTS) + 1  # Add 1 to account for last state in the trajectory
lr = 0.0003
n_epochs = 3
minibatch_size = 32
gamma = 0.99
gae_lambda = 0.95
policy_clip = 0.2
vf_coeff = 1
entropy_coeff = 0.01

SAVE_MODEL_EVERY_N_TRAINS = 30
N_TRAINING_THREADS = 4
REWARD_HISTORY_FILE = "out/reward_history.json"
OLD_AGENTS_REWARD_HISTORY_FILE = "out/old_agents_reward_history.json"


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


class Trainer:
    def __init__(self):
        self.reward_history = []
        self.old_agents_reward_history = []
        if os.path.exists(REWARD_HISTORY_FILE):
            with open(REWARD_HISTORY_FILE, "r") as f:
                self.reward_history = json.load(f)
        if os.path.exists(OLD_AGENTS_REWARD_HISTORY_FILE):
            with open(OLD_AGENTS_REWARD_HISTORY_FILE, "r") as f:
                self.old_agents_reward_history = json.load(f)

        self.n_steps = 0
        self.n_trains = 0

    async def run(self):
        self.device = init_device()

        torch.set_num_threads(N_TRAINING_THREADS)

        self.agent = Agent(
            device=self.device,
            n_actions=len(Action),
            input_shape=INPUT_SHAPE,
            gamma=gamma,
            gae_lambda=gae_lambda,
            policy_clip=policy_clip,
            minibatch_size=minibatch_size,
            n_agents=len(ACCOUNTS),
            n_epochs=n_epochs,
            alpha=lr,
            vf_coeff=vf_coeff,
            entropy_coeff=entropy_coeff,
        )
        if not self.agent.load_models():
            self.agent.save_models()

        self.workers = [
            WorkerProcess(i, account, self.remember, self.log_reward)
            for i, account in enumerate(ACCOUNTS)
        ]

        while True:
            await asyncio.sleep(10)
            for i, worker in enumerate(self.workers):
                if not worker.alive:
                    first = int(i / 2) * 2
                    second = int(i / 2) * 2 + 1
                    self.reload_worker(first)
                    self.reload_worker(second)

    def reload_worker(self, idx: int):
        self.workers[idx].reload()
        self.agent.purge_memory(idx)

    def remember(self, id, state, action, probs, vals, reward, done):
        self.agent.remember(id, state, action, probs, vals, reward, done)
        self.n_steps += 1
        if self.n_steps % horizon == 0:
            logger.warning("Training agent")
            self.agent.learn()
            self.agent.save_models()
            self.n_trains += 1
            if self.n_trains % SAVE_MODEL_EVERY_N_TRAINS == 0:
                self.agent.backup_models()

            self.inform_workers()

    def log_reward(self, data):
        if data["agent_name"]:
            self.old_agents_reward_history.append(data)
            with open(OLD_AGENTS_REWARD_HISTORY_FILE, "w") as f:
                json.dump(self.old_agents_reward_history, f)
        else:
            self.reward_history.append(data)
            with open(REWARD_HISTORY_FILE, "w") as f:
                json.dump(self.reward_history, f)

    def inform_workers(self):
        for worker in self.workers:
            worker.update_model()


if __name__ == "__main__":
    trainer = Trainer()
    asyncio.run(trainer.run())
