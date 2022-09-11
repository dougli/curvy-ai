import asyncio
import multiprocessing as mp
from queue import Empty
from typing import Callable

import torch
import torch.backends.mps

import logger
from agent import Agent
from screen import INPUT_SHAPE, Action

# Hyperparameters
horizon = 2048
lr = 0.0003
n_epochs = 10
minibatch_size = 64
gamma = 0.99
gae_lambda = 0.95
policy_clip = 0.1
vf_coeff = 1
entropy_coeff = 0.0

N_TRAINING_THREADS = 4


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


class TrainerProcess:
    def __init__(self, on_model_update: Callable):
        self.on_model_update = on_model_update

        ctx = mp.get_context("spawn")
        self.sender = ctx.Queue()
        self.receiver = ctx.Queue()
        self.process = ctx.Process(
            target=trainer_process, args=(self.receiver, self.sender)
        )
        self.process.start()

        asyncio.ensure_future(self._poll_receiver())

    def remember(self, state, action, probs, vals, reward, done):
        self.sender.put((state, action, probs, vals, reward, done))

    async def _poll_receiver(self):
        while True:
            try:
                data = self.receiver.get(block=False)
                if data["model_updated"]:
                    self.on_model_update()
            except Empty:
                await asyncio.sleep(0.25)

    def close(self):
        self.sender.close()
        self.receiver.close()
        self.process.terminate()
        self.process.join()


def trainer_process(sender: mp.Queue, receiver: mp.Queue):
    torch.set_num_threads(N_TRAINING_THREADS)

    agent = Agent(
        device=init_device(),
        n_actions=len(Action),
        input_shape=INPUT_SHAPE,
        gamma=gamma,
        gae_lambda=gae_lambda,
        policy_clip=policy_clip,
        minibatch_size=minibatch_size,
        n_agents=1,
        n_epochs=n_epochs,
        alpha=lr,
        vf_coeff=vf_coeff,
        entropy_coeff=entropy_coeff,
    )
    agent.load_models()

    n_steps = 0
    while True:
        state, action, probs, vals, reward, done = receiver.get(block=True)
        agent.remember(0, state, action, probs, vals, reward, done)
        n_steps += 1

        if n_steps % horizon == 0:
            logger.warning("learning")
            agent.learn()
            agent.save_models()
            sender.put({"model_updated": True})
