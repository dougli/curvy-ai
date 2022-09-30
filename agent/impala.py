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


class ImpalaCNN(nn.Module):
    def __init__(self, input_shape, n_actions) -> None:
        super(ImpalaCNN, self).__init__()
        self.feat_convs = []
        self.resnet1 = []
        self.resnet2 = []

        self.convs = []

        input_channels = input_shape[0]
        for num_ch in [16, 32, 32]:
            feats_convs = []
            feats_convs.append(
                nn.Conv2d(
                    in_channels=input_channels,
                    out_channels=num_ch,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            )
            # Add batch normalization -- https://arxiv.org/pdf/1812.02341.pdf --
            # Quantifying Generalization in Reinforcement Learning
            # Dramatically increases generalizability and stability
            feats_convs.append(nn.BatchNorm2d(num_ch))
            feats_convs.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            self.feat_convs.append(nn.Sequential(*feats_convs))

            input_channels = num_ch

            for i in range(2):
                resnet_block = []
                resnet_block.append(nn.ReLU())
                resnet_block.append(
                    nn.Conv2d(
                        in_channels=input_channels,
                        out_channels=num_ch,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    )
                )
                resnet_block.append(nn.BatchNorm2d(num_ch))
                resnet_block.append(nn.ReLU())
                resnet_block.append(
                    nn.Conv2d(
                        in_channels=input_channels,
                        out_channels=num_ch,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    )
                )
                resnet_block.append(nn.BatchNorm2d(num_ch))
                if i == 0:
                    self.resnet1.append(nn.Sequential(*resnet_block))
                else:
                    self.resnet2.append(nn.Sequential(*resnet_block))

        self.feat_convs = nn.ModuleList(self.feat_convs)
        self.resnet1 = nn.ModuleList(self.resnet1)
        self.resnet2 = nn.ModuleList(self.resnet2)

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(self._get_conv_out(input_shape), 256),
            nn.ReLU(),
        )
        self.actor = nn.Sequential(
            nn.Linear(256, n_actions),
            nn.Softmax(dim=-1),
        )
        self.critic = nn.Linear(256, 1)

    def _get_conv_out(self, shape):
        o = self._forward_conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x: torch.Tensor):
        x = self._forward_conv(x)
        x = self.fc(x)

        value = self.critic(x)
        dist = self.actor(x)
        dist = torch.distributions.Categorical(dist)
        return dist, value

    def _forward_conv(self, x):
        res_input = None
        for i, fconv in enumerate(self.feat_convs):
            x = fconv(x)
            res_input = x
            x = self.resnet1[i](x)
            x += res_input
            res_input = x
            x = self.resnet2[i](x)
            x += res_input
        return x

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
        if not os.path.exists(constants.BACKUP_DIR):
            os.makedirs(constants.BACKUP_DIR)
        filename = os.path.join(
            constants.BACKUP_DIR, f"model_checkpoint_{int(time.time())}"
        )
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
