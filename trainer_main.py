import asyncio
import json
import os

import torch
import torch.backends.mps
from gym.spaces import Box, Discrete

import constants
import logger
import utils
from agent.impala_feat_ext import ImpalaCNNFeat
from agent.stable_agent import StableAgent
from async_rollout_buffer import AsyncRolloutBuffer
from screen import INPUT_SHAPE, Account, Action, FakeGameEnv
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
horizon = 1024 + len(ACCOUNTS)  # Add 1 to account for last state in the trajectory
lr = lambda rem: 2.5e-4 * rem
n_epochs = 4
minibatch_size = 256
policy_clip = 0.1
vf_coeff = 0.5
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
        self.n_trains = len(utils.load_json(constants.TRAIN_LOSS_HISTORY_FILE, []))

    async def run(self):
        self.device = init_device()
        env = FakeGameEnv()

        torch.set_num_threads(N_TRAINING_THREADS)

        self.agent = StableAgent(
            "CnnPolicy",
            env=env,
            verbose=1,
            n_steps=horizon,
            policy_kwargs={"features_extractor_class": ImpalaCNNFeat},
            n_epochs=n_epochs,
            batch_size=minibatch_size,
            learning_rate=lr,
            clip_range=policy_clip,
            vf_coef=vf_coeff,
            ent_coef=entropy_coeff,
            tensorboard_log="curve-impala/logs",
        )
        if os.path.exists("curve-impala/models/ppo.zip"):
            self.agent = StableAgent.load("curve-impala/models/ppo.zip", env=env)
        self.agent.save("curve-impala/models/ppo.zip", exclude=["callback"])
        self.agent.rollout_buffer = AsyncRolloutBuffer(  # type: ignore
            self.agent.n_steps,
            env.observation_space,
            env.action_space,
            device=self.agent.device,
            gae_lambda=self.agent.gae_lambda,
            gamma=self.agent.gamma,
            n_envs=len(ACCOUNTS),
        )
        self.agent.setup_learn(total_timesteps=int(1e7), reset_num_timesteps=False)

        self.workers = [
            WorkerProcess(i, account, self.remember, self.log_reward)
            for i, account in enumerate(ACCOUNTS)
        ]

        while True:
            await asyncio.sleep(10)

    def remember(self, id, state, action, probs, vals, reward, done):
        self.agent.rollout_buffer.add(  # type: ignore
            state, action, reward, done, vals, probs, env_num=id  # type: ignore
        )
        self.agent.num_timesteps += 1
        if self.agent.num_timesteps % horizon == 0:
            logger.warning("Training agent")
            self.agent.rollout_buffer.compute_returns_and_advantage()  # type: ignore
            self.agent.callback.on_rollout_end()
            self.agent.learn()
            self.agent.save("curve-impala/models/ppo.zip", exclude=["callback"])

            if (self.agent.num_timesteps / horizon) % SAVE_MODEL_EVERY_N_TRAINS == 0:
                self.agent.save(
                    f"curve-impala/models/old/{self.agent.num_timesteps}",
                    exclude=["callback"],
                )

            self.agent.rollout_buffer.reset()  # type: ignore
            self.inform_workers()

    def log_reward(self, data):
        if data["agent_name"]:
            self.old_agents_reward_history.append(data)
            with open(OLD_AGENTS_REWARD_HISTORY_FILE, "w") as f:
                json.dump(self.old_agents_reward_history, f)
        else:
            self.agent._update_info_buffer(
                [{"episode": {"r": data["reward"], "l": data["time"]}}]
            )

    def inform_workers(self):
        for worker in self.workers:
            worker.update_model()


if __name__ == "__main__":
    trainer = Trainer()
    asyncio.run(trainer.run())
