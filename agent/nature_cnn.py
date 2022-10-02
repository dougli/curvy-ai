import os
import time
from typing import Optional

import constants
import logger
import numpy as np
import oldagents
import torch
from torch import nn

CHECKPOINT_FILE = "out/model_checkpoint"


class NatureCNN(nn.Module):
    def __init__(self, input_shape, n_actions) -> None:
        super(NatureCNN, self).__init__()

        self.shared = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.shared_middleware = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
        )

        self.actor = nn.Sequential(
            nn.Linear(512, n_actions),
            nn.Softmax(dim=-1),
        )
        self.critic = nn.Sequential(
            nn.Linear(512, 1),
        )

    def _get_conv_out(self, shape):
        o = self.shared(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        x = self.shared(x)
        x = self.shared_middleware(x)
        value = self.critic(x)
        dist = self.actor(x)
        dist = torch.distributions.Categorical(dist)
        return dist, value

    def choose_action(self, state):
        dist, value = self(state)
        action = dist.sample()

        probs = torch.squeeze(dist.log_prob(action)).item()
        action = torch.squeeze(action).item()
        value = torch.squeeze(value).item()

        return action, probs, value

    def save_checkpoint(self):
        filename = os.path.join(os.path.dirname(__file__), "..", CHECKPOINT_FILE)
        logger.success(f"Saving checkpoint to {filename}")
        torch.save(self.state_dict(), filename)

    def load_checkpoint(self):
        filename = os.path.join(os.path.dirname(__file__), "..", CHECKPOINT_FILE)
        if os.path.exists(filename):
            logger.success(f"Loading checkpoint from {filename}")
            self.load_state_dict(torch.load(filename))
            return True
        else:
            logger.warning(f"Checkpoint file {filename} not found")
            return False

    def backup_checkpoint(self):
        directory = os.path.join(os.path.dirname(__file__), "..", constants.BACKUP_DIR)
        if not os.path.exists(directory):
            os.makedirs(directory)
        filename = os.path.join(directory, f"model_checkpoint_{int(time.time())}")
        logger.success(f"Backing up checkpoint to {filename}")
        torch.save(self.state_dict(), filename)

    def load_random_backup(self) -> Optional[str]:
        filename = oldagents.select_old_agent()
        if not filename:
            logger.warning(f"No backups found in {constants.BACKUP_DIR}")
            return None
        filepath = os.path.join(constants.BACKUP_DIR, filename)
        logger.success(f"Loading backup from {filename}")
        self.load_state_dict(torch.load(filepath))
        return filename
