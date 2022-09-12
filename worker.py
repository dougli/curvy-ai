import asyncio
import multiprocessing as mp
import random
import time
from queue import Empty
from typing import Callable

import torch

import logger
from agent import CurvyNet
from screen import INPUT_SHAPE, Account, Action, Game

N_GAME_THREADS = 2
CURVE_FEVER = "https://curvefever.pro"

RANDOM_OLD_MODEL = 0.2


class WorkerProcess:
    def __init__(
        self,
        id: int,
        account: Account,
        on_remember: Callable,
        on_log_reward: Callable,
    ):
        self.id = id
        self.account = account
        self._on_rememeber = on_remember
        self._on_log_reward = on_log_reward

        ctx = mp.get_context("spawn")
        self.sender = ctx.Queue()
        self.receiver = ctx.Queue()
        self.process = ctx.Process(
            target=start_worker_process,
            args=(
                self.account,
                self.receiver,
                self.sender,
            ),
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
                        self._on_log_reward(message["data"], message["old_agent_name"])
            except Empty:
                await asyncio.sleep(0.1)

    def update_model(self):
        self.sender.put({"type": "update_model"})


class Worker:
    def __init__(
        self,
        account: Account,
        sender: mp.Queue,
        receiver: mp.Queue,
    ):
        self.account = account
        self.sender = sender
        self.receiver = receiver

        torch.set_num_threads(N_GAME_THREADS)

        self.cpu = torch.device("cpu")

        self.model = CurvyNet((1, *INPUT_SHAPE), len(Action)).to(self.cpu)
        self.model.load_checkpoint()

        self.model_lock = mp.Lock()

        self.model_name = None
        self.using_old_model = False

    async def run(self):
        game = Game(
            self.account.match_name,
            self.account.match_password,
            show_play_area=self.account.is_host,
        )
        await game.launch(CURVE_FEVER)
        await game.login(self.account.email, self.account.password)
        if self.account.is_host:
            await game.create_match()
        else:
            await game.join_match()

        asyncio.ensure_future(self._poll_receiver())

        n_steps = 0
        episode = 0
        for _ in range(1000000):
            if not game.in_play:
                if episode > 0:
                    if self.using_old_model:
                        self.log_reward({"score": game.score, "steps": n_steps})
                    else:
                        self.log_reward(n_steps)
                    logger.info(
                        f"Episode {episode}, steps: {n_steps}, "
                        f"game score {game.score}, fps {game.fps}"
                    )

                self.model_lock.acquire()
                if random.random() < RANDOM_OLD_MODEL:
                    self.using_old_model = True
                    self.model_name = self.model.load_random_backup()
                elif self.using_old_model:
                    self.using_old_model = False
                    self.model.load_checkpoint()
                    self.model_name = None
                self.model_lock.release()

                await game.skip_powerup()
                if self.account.is_host:
                    for username in self.account.wait_for:
                        await game.wait_for_player_ready(username)
                    await game.start_match()
                await game.wait_for_start()
                n_steps = 0
                episode += 1

            ready_to_play = await game.wait_for_alive()
            if not ready_to_play:
                continue

            done = False
            n_steps_inner = 0
            start_inner = time.time()
            while not done:
                state = game.play_area
                torch_state = torch.tensor([state], dtype=torch.float32).to(self.cpu)
                action, prob, value = self.model.choose_action(torch_state)
                reward, done = await game.step(Action(action))
                n_steps += 1
                n_steps_inner += 1
                if not self.using_old_model:
                    self.remember(state, action, prob, value, reward, done)
            logger.error(
                f"Inner steps: {n_steps_inner}, time: {time.time() - start_inner}, FPS: {n_steps_inner / (time.time() - start_inner)}"
            )

    def remember(self, state, action, prob, value, reward, done):
        self.sender.put(
            {
                "type": "remember",
                "data": (state, action, prob, value, reward, done),
            }
        )

    def log_reward(self, reward):
        self.sender.put(
            {"type": "log_reward", "data": reward, "old_agent_name": self.model_name}
        )

    async def _poll_receiver(self):
        while True:
            try:
                data = self.receiver.get(block=False)
                if data["type"] == "update_model":
                    self.model_lock.acquire()
                    if not self.using_old_model:
                        self.model.load_checkpoint()
                    self.model_lock.release()
            except Empty:
                await asyncio.sleep(0.25)


def start_worker_process(*args, **kwargs):
    worker = Worker(*args, **kwargs)
    asyncio.run(worker.run())
