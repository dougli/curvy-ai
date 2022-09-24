import asyncio
import multiprocessing as mp
import random
import time
from queue import Empty
from typing import Callable

import numpy as np
import torch

import logger
from agent import ImpalaCNN
from screen import INPUT_SHAPE, Account, Action, Game

N_GAME_THREADS = 2
CURVE_FEVER = "https://curvefever.pro"

RANDOM_OLD_MODEL = 0.1

MAX_IDLE_TIME = 60  # 1 minute
INITIAL_IDLE_TIME = 60 * 2  # 2 minutes

INITIAL_SLEEP = 2.75


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

        self.last_message = time.time() + INITIAL_IDLE_TIME
        self.process.start()
        asyncio.ensure_future(self._poll_receiver())

    @property
    def alive(self):
        return (
            self.last_message > time.time() - MAX_IDLE_TIME and self.process.is_alive()
        )

    async def _poll_receiver(self):
        while True:
            try:
                while True:
                    message = self.receiver.get(block=False)
                    if message["type"] == "remember":
                        self._on_rememeber(self.id, *message["data"])
                    elif message["type"] == "log_reward":
                        self._on_log_reward(message["data"])
                    self.last_message = time.time()
            except Empty:
                await asyncio.sleep(0.1)

    def update_model(self):
        self.sender.put({"type": "update_model"})

    def reload(self):
        logger.error(f"Reloading worker {self.id}")
        self.sender.put({"type": "reload"})
        self.last_message = time.time() + INITIAL_IDLE_TIME


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

        self.model = ImpalaCNN((2, *INPUT_SHAPE), len(Action)).to(self.cpu)
        self.model.load_checkpoint()

        self.model_lock = mp.Lock()
        self.old_model_name = None

        self.game = Game(
            self.account.match_name,
            self.account.match_password,
            show_screen=True,
        )

    async def run(self):
        await self.game.launch(CURVE_FEVER)
        await self.game.login(self.account.email, self.account.password)

        asyncio.ensure_future(self._poll_receiver())

        await self.main_loop()

    async def main_loop(self):
        if self.account.is_host:
            await self.game.create_match()
        else:
            await self.game.join_match()

        just_started = True
        for _ in range(1000000):
            if not self.game.in_play:
                await self.game.skip_powerup()
                if self.account.is_host:
                    for username in self.account.wait_for:
                        await self.game.wait_for_player_ready(username)
                    await self.game.start_match()
                await self.game.wait_for_start()
                just_started = True

            self.model_lock.acquire()
            if random.random() < RANDOM_OLD_MODEL:
                self.old_model_name = self.model.load_random_backup()
            elif self.old_model_name:
                self.model.load_checkpoint()
                self.old_model_name = None
            self.model_lock.release()

            ready_to_play = await self.game.wait_for_alive()
            if not ready_to_play:
                continue
            if just_started:
                # Sleep for 6 seconds because the there's some initial splash screen
                # when initially joining a match
                await asyncio.sleep(6)

            await asyncio.sleep(INITIAL_SLEEP)
            self.game.update_last_reward_time()

            done = False
            n_steps_inner = 0
            start_time = time.time()
            round_reward = 0
            prev_state = np.zeros((1, *INPUT_SHAPE))
            won = False
            while not done:
                state = self.game.play_area
                stacked = np.concatenate([prev_state, state], axis=0)
                torch_state = torch.tensor([stacked], dtype=torch.float32).to(self.cpu)
                action, prob, value = self.model.choose_action(torch_state)
                reward, done, won = await self.game.step(Action(action))
                n_steps_inner += 1
                round_reward += reward
                if not self.old_model_name:
                    self.remember(stacked, action, prob, value, reward, done)
                prev_state = state
            elapsed = time.time() - start_time
            logger.train_info(
                f"round reward: {round_reward}, time: {elapsed}, FPS: {n_steps_inner / elapsed}"
            )
            self.log_reward(
                {
                    "won": won,
                    "reward": round_reward,
                    "time": elapsed,
                    "agent_name": self.old_model_name,
                }
            )
            just_started = False
            # Wait 1.5 seconds after the round ends as the UI is still showing the
            # end-of-round scoreboard
            await asyncio.sleep(1.5)

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
                    self.model_lock.acquire()
                    if not self.old_model_name:
                        self.model.load_checkpoint()
                    self.model_lock.release()
                elif data["type"] == "reload":
                    # self.game.on_reload()
                    raise NotImplementedError()
            except Empty:
                await asyncio.sleep(0.25)


def start_worker_process(*args, **kwargs):
    worker = Worker(*args, **kwargs)
    asyncio.run(worker.run())
