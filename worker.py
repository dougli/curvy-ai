import asyncio
import multiprocessing as mp
from queue import Empty
from typing import Callable

import torch

import logger
from agent import CurvyNet
from screen import INPUT_SHAPE, Action, Game

N_GAME_THREADS = 4
CURVE_FEVER = "https://curvefever.pro"


class WorkerProcess:
    def __init__(
        self,
        id: int,
        email: str,
        password: str,
        on_remember: Callable,
        on_log_reward: Callable,
    ):
        self.id = id
        self.email = email
        self.password = password
        self._on_rememeber = on_remember
        self._on_log_reward = on_log_reward

        ctx = mp.get_context("spawn")
        self.sender = ctx.Queue()
        self.receiver = ctx.Queue()
        self.process = ctx.Process(
            target=start_worker_process,
            args=(self.email, self.password, self.receiver, self.sender),
        )
        self.process.start()

        asyncio.ensure_future(self._poll_receiver())

    async def _poll_receiver(self):
        while True:
            try:
                while True:
                    message = self.receiver.get(block=False)
                    if message["type"] == "remember":
                        self._on_rememeber(self.id, *message["data"])
                    elif message["type"] == "log_reward":
                        self._on_log_reward(message["data"])
            except Empty:
                await asyncio.sleep(0.1)

    def update_model(self):
        self.sender.put({"type": "update_model"})


class Worker:
    def __init__(self, email: str, password: str, sender: mp.Queue, receiver: mp.Queue):
        self.email = email
        self.password = password
        self.sender = sender
        self.receiver = receiver

        torch.set_num_threads(N_GAME_THREADS)

        self.cpu = torch.device("cpu")

        self.model = CurvyNet((1, *INPUT_SHAPE), len(Action)).to(self.cpu)
        self.model.load_checkpoint()

    async def run(self):
        game = Game(self.email, self.password)
        await game.launch(CURVE_FEVER)
        await game.login()
        await game.create_match()
        await game.start_match()

        asyncio.ensure_future(self._poll_receiver())

        n_steps = 0
        avg_score = 0

        for episode in range(10000):
            await game.wait_for_alive()

            total_reward = 0
            done = False
            while not done:
                state = game.play_area
                torch_state = torch.tensor([state], dtype=torch.float32).to(self.cpu)
                action, prob, value = self.model.choose_action(torch_state)
                reward, done = await game.step(Action(action))
                n_steps += 1
                total_reward += reward
                self.remember(state, action, prob, value, reward, done)

            self.log_reward(total_reward)

            logger.info(
                f"Episode {episode}, reward {round(total_reward, 3)}, avg {round(avg_score, 3)}, steps {n_steps}, game score {game.player.score}"
            )

    def remember(self, state, action, prob, value, reward, done):
        self.sender.put(
            {
                "type": "remember",
                "data": (state, action, prob, value, reward, done),
            }
        )

    def log_reward(self, reward):
        self.sender.put({"type": "log_reward", "data": reward})

    async def _poll_receiver(self):
        while True:
            try:
                data = self.receiver.get(block=False)
                if data["type"] == "update_model":
                    self.model.load_checkpoint()
            except Empty:
                await asyncio.sleep(0.25)


def start_worker_process(*args, **kwargs):
    worker = Worker(*args, **kwargs)
    asyncio.run(worker.run())
