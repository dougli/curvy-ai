import copy
from typing import Literal

import numpy as np
import torch
from torch import nn


# Double Deep Q-Learning
class CurvyNet(nn.Module):
    def __init__(self, input_shape, n_actions) -> None:
        super(CurvyNet, self).__init__()
        self.online_conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.online_fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

        self.target_conv = copy.deepcopy(self.online_conv)
        self.target_fc = copy.deepcopy(self.online_fc)

        for p in self.target_conv.parameters():
            p.requires_grad = False
        for p in self.target_fc.parameters():
            p.requires_grad = False

    def _get_conv_out(self, shape):
        o = self.online_conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x, model=Literal["online", "target"]):
        if model == "online":
            conv_out = self.online_conv(x).view(x.size()[0], -1)
            return self.online_fc(conv_out)
        elif model == "target":
            conv_out = self.target_conv(x).view(x.size()[0], -1)
            return self.target_fc(conv_out)
