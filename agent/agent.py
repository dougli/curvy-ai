import os

import constants
import logger
import numpy as np
import torch as T
import torch.optim as optim
import utils

from agent.net import CurvyNet

OPTIMIZER_FILE = "out/optimizer.pt"


class PPOMemory:
    def __init__(self, gamma: float, gae_lambda: float):
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

    def store(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []

    def compute_gae(self):
        # Calculate advantages. We're throwing away the last value because we don't have
        # a next state
        n_steps = len(self.rewards) - 1

        advantages = np.zeros(n_steps)
        gae = 0
        for t in reversed(range(n_steps)):
            non_terminal = 1 - int(self.dones[t])
            delta = (
                self.rewards[t]
                + self.gamma * self.vals[t + 1] * non_terminal
                - self.vals[t]
            )
            gae = delta + self.gamma * self.gae_lambda * non_terminal * gae
            advantages[t] = gae

        return advantages

    def export_for_learning(self):
        if len(self.rewards) <= 1:
            self.clear()
            return None

        result = (
            self.states[:-1],
            self.actions[:-1],
            self.probs[:-1],
            self.vals[:-1],
            self.compute_gae(),
        )
        self.clear()
        return result


class Agent:
    def __init__(
        self,
        *,
        device: T.device,
        n_actions,
        input_shape,
        gamma: float,
        alpha: float,
        gae_lambda: float,
        policy_clip: float,
        minibatch_size: int,
        n_agents: int,
        n_epochs: int,
        vf_coeff: float,
        entropy_coeff: float,
    ):
        self.policy_clip = policy_clip
        self.minibatch_size = minibatch_size
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.vf_coeff = vf_coeff
        self.entropy_coeff = entropy_coeff

        self.model = CurvyNet((2, *input_shape), n_actions).to(device)
        self.memories = [PPOMemory(gamma, gae_lambda) for _ in range(n_agents)]
        self.device = device

        self.optimizer = optim.Adam(self.model.parameters(), lr=alpha)
        if os.path.exists(OPTIMIZER_FILE):
            logger.success("Loading optimizer from file")
            self.optimizer.load_state_dict(T.load(OPTIMIZER_FILE))
        else:
            logger.warning("Optimizer file not found. Starting from scratch")

    def purge_memory(self, env_idx):
        self.memories[env_idx].clear()

    def remember(self, env_idx, state, action, probs, vals, reward, done):
        self.memories[env_idx].store(state, action, probs, vals, reward, done)

    def save_models(self):
        self.model.save_checkpoint()

    def backup_models(self):
        self.model.backup_checkpoint()

    def load_models(self) -> bool:
        return self.model.load_checkpoint()

    def learn(self):
        memories = [m.export_for_learning() for m in self.memories]

        state_arr = np.concatenate([m[0] for m in memories if m])
        action_arr = np.concatenate([m[1] for m in memories if m])
        old_prob_arr = np.concatenate([m[2] for m in memories if m])
        vals_arr = np.concatenate([m[3] for m in memories if m])
        advantage = np.concatenate([m[4] for m in memories if m])
        advantage = T.tensor(advantage, dtype=T.float32).to(self.device)

        indices = np.arange(state_arr.shape[0])

        n_updates = 0
        mean_actor_loss = 0
        mean_critic_loss = 0
        mean_entropy_loss = 0

        for _ in range(self.n_epochs):
            np.random.shuffle(indices)
            for start in range(0, len(state_arr), self.minibatch_size):
                end = start + self.minibatch_size
                if end > len(state_arr):
                    continue

                batch = indices[start:end]

                states = T.tensor(state_arr[batch], dtype=T.float32).to(self.device)
                old_probs = T.tensor(old_prob_arr[batch], dtype=T.float32).to(
                    self.device
                )
                actions = T.tensor(action_arr[batch], dtype=T.float32).to(self.device)
                values = T.tensor(vals_arr[batch], dtype=T.float32).to(self.device)

                dist, critic_value = self.model(states)

                critic_value = T.squeeze(critic_value)

                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()
                # prob_ratio = (new_probs - old_probs).exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = (
                    T.clamp(prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip)
                    * advantage[batch]
                )
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()

                entropy = dist.entropy().mean()

                returns = advantage[batch] + values
                critic_loss = (returns - critic_value) ** 2
                critic_loss = critic_loss.mean()

                total_loss = (
                    actor_loss
                    + self.vf_coeff * critic_loss
                    - self.entropy_coeff * entropy
                )
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                n_updates += 1
                mean_actor_loss += actor_loss.item()
                mean_critic_loss += critic_loss.item()
                mean_entropy_loss += entropy.item()
        T.save(self.optimizer.state_dict(), OPTIMIZER_FILE)
        self.save_loss_history(
            {
                "actor_loss": mean_actor_loss / n_updates,
                "critic_loss": mean_critic_loss / n_updates,
                "entropy_loss": mean_entropy_loss / n_updates,
            }
        )

    def save_loss_history(self, loss_history):
        # Read the loss history file first
        data = utils.load_json(constants.TRAIN_LOSS_HISTORY_FILE, [])

        # Save the loss history
        data.append(loss_history)
        utils.save_json(constants.TRAIN_LOSS_HISTORY_FILE, data)
        logger.warning(f"Loss: {loss_history}")
