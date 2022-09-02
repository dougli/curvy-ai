import asyncio
import random
import time

import logger
from screen import Action, Game


async def main():
    game = Game("lidouglas@gmail.com", "mzk-drw-krd3EVP5axn", show_screen=True)
    await game.launch()
    await game.login()
    await game.create_match()
    await game.start_match()

    while True:
        await game.wait_for_alive()

        total_reward = 0
        done = False
        while not done:
            action = random.randint(0, 2)

            start_time = time.time()
            reward, done = await game.step(Action(action))
            end_time = time.time()
            logger.warning(
                f"Action: {Action(action).name}, Time: {end_time - start_time}"
            )

            total_reward += reward

            # Log what's going on
            player = game.player
            fps = game.fps
            logger.info(
                f"Score: {player.score}, Alive: {player.alive}, Dead: {player.dead}, FPS: {round(fps)}, Total Reward: {total_reward}"
            )


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
