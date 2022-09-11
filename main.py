import asyncio
import json
import os
from multiprocessing import freeze_support

import numpy as np
import torch

import logger
from agent import CurvyNet
from screen import INPUT_SHAPE, Action, Game
from trainer import TrainerProcess

CURVE_FEVER = "https://curvefever.pro"
WEB_GL_GAME = "https://playcanv.as/p/LwskqxXT/"

REWARD_HISTORY_FILE = "out/reward_history.json"

N_GAME_THREADS = 4


async def main():
    cpu = torch.device("cpu")
    torch.set_num_threads(N_GAME_THREADS)

    model = CurvyNet((1, *INPUT_SHAPE), len(Action)).to(cpu)
    model.load_checkpoint()

    trainer = TrainerProcess(on_model_update=model.load_checkpoint)

    game = Game("hypermoop's match", "pwei93Fx")
    await game.launch(CURVE_FEVER)
    await game.login("lidouglas@gmail.com", "mzk-drw-krd3EVP5axn")
    await game.create_match()
    await game.start_match()
    # await game.launch(WEB_GL_GAME)

    if os.path.exists(REWARD_HISTORY_FILE):
        with open(REWARD_HISTORY_FILE, "r") as f:
            reward_history = json.load(f)
    else:
        reward_history = []

    n_steps = 0
    avg_score = 0

    for episode in range(10000):
        await game.wait_for_alive()

        total_reward = 0
        done = False
        while not done:
            state = game.play_area
            torch_state = torch.tensor([state], dtype=torch.float32).to(cpu)
            action, prob, value = model.choose_action(torch_state)
            reward, done = await game.step(Action(action))
            n_steps += 1
            total_reward += reward
            trainer.remember(state, action, prob, value, reward, done)

        reward_history.append(total_reward)
        avg_score = np.mean(reward_history[-100:])

        logger.info(
            f"Episode {episode}, reward {round(total_reward, 3)}, avg {round(avg_score, 3)}, steps {n_steps}, game score {game.player.score}"
        )

        with open(REWARD_HISTORY_FILE, "w") as f:
            json.dump(reward_history, f)


if __name__ == "__main__":
    freeze_support()
    asyncio.run(main())
