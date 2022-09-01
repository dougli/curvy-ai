import asyncio
import random
import time

import logger
from screen import Game
from screen.types import Action

# frames = 0
# start_time = time.time()
# while True:
#     screen.frame()
#     image = screen.game_area()
#     game_state = screen.game_state()
#     now = time.time()
#     frames += 1
#     fps = round(frames / (now - start_time), 2)
#     print(
#         f"Score: {game_state.score}, Alive: {game_state.alive}, Dead: {game_state.dead}, FPS: {fps}"
#     )


async def main():
    game = Game("lidouglas@gmail.com", "mzk-drw-krd3EVP5axn", show_screen=True)
    await game.launch()
    await game.start_game()
    while True:
        state = game.state
        fps = game.fps
        logger.info(
            f"Score: {state.score}, Alive: {state.alive}, Dead: {state.dead}, FPS: {fps}"
        )
        if state.alive:
            action = random.randint(0, 4)
            start_time = time.time()
            await game.set_action(Action(action))
            end_time = time.time()
            logger.warning(
                f"Action: {Action(action).name}, Time: {end_time - start_time}"
            )
        await asyncio.sleep(1 / 60)

    await game.close()


asyncio.run(main())

# env = gym.make("CartPole-v1", new_step_api=True)


# def init_device():
#     cuda = torch.cuda.is_available()  # Nvidia GPU
#     mps = torch.backends.mps.is_available()  # M1 processors
#     if cuda:
#         print("Using CUDA")
#         device = torch.device("cuda")
#     # elif mps:
#     #     print("Using MPS")
#     #     device = torch.device("mps")
#     else:
#         print("Using CPU")
#         device = torch.device("cpu")
#     return device


# T = 20


# def run(device):
#     iteration = 0
#     state = env.reset()
#     next_state = None
#     total_reward = 0
#     steps = 0
#     learned_steps = 0

#     while iteration < 10000:
#         state = torch.FloatTensor(state).unsqueeze(0).to(device)
#         dist, value = agent.act(state)
#         action = dist.sample()
#         next_state, reward, term, trunc, info = env.step(action.item())  # type: ignore
#         done = bool(term or trunc)
#         agent.remember(state, action, dist.log_prob(action), value, reward, done)
#         total_reward += reward
#         state = env.reset() if done else next_state

#         steps += 1

#         if done:
#             print(
#                 f"Iteration: {iteration}, Reward: {total_reward}, Learning steps: {learned_steps}"
#             )
#             iteration += 1
#             total_reward = 0

#         if steps % T == 0:
#             next_state = torch.FloatTensor(next_state).unsqueeze(0).to(device)
#             agent.learn(next_state)
#             steps = 0
#             learned_steps += 1


# device = init_device()
# model = CurvyNet(4, 24, 2)
# model.to(device)
# agent = Agent(model, device)
# run(device)
