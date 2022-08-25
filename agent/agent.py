import random
from collections import deque

import numpy as np
import torch

from agent.net import CurvyNet


class Agent:
    def __init__(self, state_space, action_space, save_dir):
        self.state_space = state_space
        self.action_space = action_space
        self.save_dir = save_dir
        self.model = CurvyNet(self.state_space, self.action_space)
        self.model.load_state_dict(torch.load(self.save_dir))
        self.model.eval()

        self.net = CurvyNet(self.state_space, self.action_space).float()

        self.exploration_rate = 1.0
        self.exploration_rate_decay = 0.99999975
        self.exploration_rate_min = 0.1
        self.curr_step = 0

        self.memory = deque(maxlen=30000)
        self.batch_size = 32

        self.remember_every = 5  # How often to remember. We just don't have emough memory to remember every step.
        self.save_every = 9000  # Save every 9000 steps -- roughly 5 minutes at 30 fps

        self.gamma = 0.9  # Discount factor
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00025)
        self.loss_fn = torch.nn.SmoothL1Loss()

    def act(self, state: np.ndarray):
        # Exploration vs exploitation
        if np.random.rand() < self.exploration_rate:
            return np.random.randint(self.action_space)
        else:
            state_tensor = torch.from_numpy(state)
            action_values = self.net(state_tensor, model="online")
            action_idx = torch.argmax(action_values).item()

        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate, self.exploration_rate_min)

        self.curr_step += 1
        return action_idx

    def remember(
        self,
        state: np.ndarray,
        next_state: np.ndarray,
        action: int,
        reward: float,
        done: bool,
    ):
        if self.curr_step % self.remember_every == 0:
            entry = (
                torch.from_numpy(state).float(),
                torch.from_numpy(next_state).float(),
                torch.tensor([action]).float(),
                torch.tensor([reward]).float(),
                torch.tensor([done]).float(),
            )
            self.memory.append(entry)

    def recall(self):
        batch = random.sample(self.memory, self.batch_size)
        state, next_state, action, reward, done = map(torch.stack, zip(*batch))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

    def td_estimate(self, state, action):
        current_Q = self.net(state, model="online")[
            np.arange(0, self.batch_size), action
        ]  # Q_online(s,a)
        return current_Q

    def td_target(self, reward, next_state, done):
        next_state_Q = self.net(next_state, model="online")
        best_action = torch.argmax(next_state_Q, dim=1)
        next_Q = self.net(next_state, model="target")[
            np.arange(0, self.batch_size), best_action
        ]
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()

    def update_Q_online(self, td_estimate, td_target):
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_Q_target(self):
        self.net.target_conv.load_state_dict(self.net.online_conv.state_dict())
        self.net.target_fc.load_state_dict(self.net.online_fc.state_dict())
