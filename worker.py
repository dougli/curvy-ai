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

KILL_AFTER_N_SECONDS = 60 * 3  # 3 minutes


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

        self.to_kill = False
        self.last_message = time.time()
        self.process.start()
        asyncio.ensure_future(self._poll_receiver())

    @property
    def alive(self):
        return self.process.is_alive()

    def kill(self):
        self.to_kill = True

    async def _poll_receiver(self):
        while True:
            if self.to_kill or time.time() - self.last_message > KILL_AFTER_N_SECONDS:
                logger.error(f"Killing worker {self.id}")
                self.process.terminate()
                await asyncio.sleep(5)
                self.process.kill()
                self.process.join()
                self.process.close()
                return

            try:
                while True:
                    message = self.receiver.get(block=False)
                    if message["type"] == "remember":
                        self._on_rememeber(self.id, *message["data"])
                    elif message["type"] == "log_reward":
                        self._on_log_reward(message["data"], message["old_agent_name"])
                    elif message["type"] == "heartbeat":
                        pass
                    self.last_message = time.time()
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
        self.old_model_name = None

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

        episode = 0
        total_reward = 0
        round_time = 0
        for _ in range(1000000):
            if not game.in_play:
                if episode > 0:
                    if self.old_model_name:
                        self.log_reward(
                            {"score": game.score, "total_reward": total_reward}
                        )
                    else:
                        self.log_reward(round_time)
                    logger.train_info(
                        f"Episode {episode}, total_round_time {round_time}, "
                        f"total_reward {total_reward}, "
                        f"game score {game.score}, fps {game.fps}"
                    )

                self.model_lock.acquire()
                if random.random() < RANDOM_OLD_MODEL:
                    self.old_model_name = self.model.load_random_backup()
                elif self.old_model_name:
                    self.model.load_checkpoint()
                    self.old_model_name = None
                self.model_lock.release()

                await game.skip_powerup()
                if self.account.is_host:
                    for username in self.account.wait_for:
                        await game.wait_for_player_ready(username)
                    await game.start_match()
                await game.wait_for_start()
                episode += 1
                round_time = 0
                total_reward = 0

            ready_to_play = await game.wait_for_alive()
            if not ready_to_play:
                continue
            if round_time == 0:
                # Sleep for 5 seconds because the there's some initial splash screen
                # when initially joining a match
                await asyncio.sleep(5)

            done = False
            n_steps_inner = 0
            start_round_time = time.time()
            round_reward = 0
            while not done:
                state = game.play_area
                torch_state = torch.tensor([state], dtype=torch.float32).to(self.cpu)
                action, prob, value = self.model.choose_action(torch_state)
                reward, done = await game.step(Action(action))
                n_steps_inner += 1
                round_reward += reward
                if not self.old_model_name:
                    self.remember(state, action, prob, value, reward, done)
            logger.train_info(
                f"round reward: {round_reward}, time: {time.time() - start_round_time}, FPS: {n_steps_inner / (time.time() - start_round_time)}"
            )
            round_time += time.time() - start_round_time
            total_reward += round_reward
            self.heartbeat()

    def remember(self, state, action, prob, value, reward, done):
        self.sender.put(
            {
                "type": "remember",
                "data": (state, action, prob, value, reward, done),
            }
        )

    def log_reward(self, reward):
        self.sender.put(
            {
                "type": "log_reward",
                "data": reward,
                "old_agent_name": self.old_model_name,
            }
        )

    def heartbeat(self):
        self.sender.put({"type": "heartbeat"})

    async def _poll_receiver(self):
        while True:
            try:
                data = self.receiver.get(block=False)
                if data["type"] == "update_model":
                    self.model_lock.acquire()
                    if not self.old_model_name:
                        self.model.load_checkpoint()
                    self.model_lock.release()
            except Empty:
                await asyncio.sleep(0.25)


def start_worker_process(*args, **kwargs):
    worker = Worker(*args, **kwargs)
    asyncio.run(worker.run())
