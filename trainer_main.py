import asyncio
import json
import os

import torch
import torch.backends.mps

import logger
from agent import Agent
from screen import INPUT_SHAPE, Account, Action
from worker import WorkerProcess

# Hyperparameters
horizon = 256 * 8
lr = 0.0003
n_epochs = 10
minibatch_size = 64
gamma = 0.99
gae_lambda = 0.95
policy_clip = 0.1
vf_coeff = 1
entropy_coeff = 0.0

ACCOUNTS = [
    Account(
        "lidouglas@gmail.com",
        "mzk-drw-krd3EVP5axn",
        "hypermoop's match",
        "vQgJ3zaP",
        True,
        ("eafdorf",),
    ),
    Account(
        "wheels.hallow_0g@icloud.com",
        "8.h*oVJokU69.GuHyLMf",
        "hypermoop's match",
        "vQgJ3zaP",
        False,
    ),
    # Account(
    #     "misstep.gills.0x@icloud.com",
    #     ".-.qqn@F@64eiwYk*Mao",
    #     "aieegame",
    #     "18xielPa",
    #     True,
    #     ("coiifan",),
    # ),
    # Account(
    #     "woyexo5962@iunicus.com", "RQZ1tgt3etg3wfk-cmu", "aieegame", "18xielPa", False
    # ),
]


N_TRAINING_THREADS = 4
REWARD_HISTORY_FILE = "out/reward_history.json"


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
        if os.path.exists(REWARD_HISTORY_FILE):
            with open(REWARD_HISTORY_FILE, "r") as f:
                self.reward_history = json.load(f)

        self.n_steps = 0

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
        self.agent.load_models()

        self.workers = [
            WorkerProcess(i, account, self.remember, self.log_reward)
            for i, account in enumerate(ACCOUNTS)
        ]

        while True:
            await asyncio.sleep(100)

    def remember(self, id, state, action, probs, vals, reward, done):
        self.agent.remember(id, state, action, probs, vals, reward, done)
        self.n_steps += 1
        if self.n_steps % horizon == 0:
            logger.warning("Training agent")
            self.agent.learn()
            self.agent.save_models()
            self.inform_workers()

    def log_reward(self, reward):
        self.reward_history.append(reward)
        # with open(REWARD_HISTORY_FILE, "w") as f:
        #     json.dump(self.reward_history, f)

    def inform_workers(self):
        for worker in self.workers:
            worker.update_model()


if __name__ == "__main__":
    trainer = Trainer()
    asyncio.run(trainer.run())
