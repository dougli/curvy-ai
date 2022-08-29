import numpy as np
import torch
from torch import nn


# Proximal Policy Optimization (PPO)
class CurvyNet(nn.Module):
    def __init__(self, input_shape, hidden_units, n_actions) -> None:
        super(CurvyNet, self).__init__()

        # self.shared = nn.Sequential(
        #     nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=3, stride=2),
        #     nn.Conv2d(32, 64, kernel_size=4, stride=2),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 64, kernel_size=3, stride=1),
        #     nn.ReLU(),
        # )
        # conv_out_size = self._get_conv_out(input_shape)
        # self.actor = nn.Sequential(
        #     nn.Linear(conv_out_size, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, n_actions),
        #     nn.Softmax(),
        # )
        # self.critic = nn.Sequential(
        #     nn.Linear(conv_out_size, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 1),
        # )

        # ----------------------------------------------------------------------
        # Cartpole model
        # ----------------------------------------------------------------------
        # self.shared = nn.Sequential(
        #     nn.Linear(input_shape, hidden_units),
        #     nn.ReLU(),
        #     nn.Linear(hidden_units, hidden_units),
        #     nn.ReLU(),
        # )

        self.actor = nn.Sequential(
            nn.Linear(input_shape, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, n_actions),
            nn.Softmax(dim=-1),
        )
        self.critic = nn.Sequential(
            nn.Linear(input_shape, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, 1),
        )

    # def _get_conv_out(self, shape):
    #     o = self.shared(torch.zeros(1, *shape))
    #     return int(np.prod(o.size()))

    def forward(self, x):
        # x = self.shared(x)
        value = self.critic(x)
        action_probs = self.actor(x)
        probs = torch.distributions.Categorical(action_probs)
        return probs, value
