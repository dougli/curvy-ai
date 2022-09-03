import time

import numpy as np
import torch as T
import torch.optim as optim

from agent.net import CurvyNet


class PPOMemory:
    def __init__(self, minibatch_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = minibatch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int32)
        np.random.shuffle(indices)
        batches = [indices[i : i + self.batch_size] for i in batch_start]

        return (
            np.array(self.states),
            np.array(self.actions),
            np.array(self.probs),
            np.array(self.vals),
            np.array(self.rewards),
            np.array(self.dones),
            batches,
        )

    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []


class Agent:
    def __init__(
        self,
        *,
        device: T.device,
        n_actions,
        input_shape,
        gamma=0.99,
        alpha=0.0003,
        gae_lambda=0.95,
        policy_clip=0.2,
        minibatch_size=64,
        n_epochs=10,
        vf_coeff=0.5,
        entropy_coeff=0.01,
    ):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.vf_coeff = vf_coeff
        self.entropy_coeff = entropy_coeff

        self.model = CurvyNet((1, *input_shape), n_actions).to(device)
        self.memory = PPOMemory(minibatch_size)
        self.device = device

        self.optimizer = optim.Adam(self.model.parameters(), lr=alpha)

    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def save_models(self):
        print("Saving models...")
        self.model.save_checkpoint()

    def load_models(self):
        print("... loading models ...")
        self.model.load_checkpoint()

    def choose_action(self, observation):
        state = T.tensor([observation], dtype=T.float32).to(self.device)

        dist, value = self.model(state)
        action = dist.sample()

        probs = T.squeeze(dist.log_prob(action)).item()
        action = T.squeeze(action).item()
        value = T.squeeze(value).item()

        return action, probs, value

    def learn(self):
        for _ in range(self.n_epochs):
            (
                state_arr,
                action_arr,
                old_prob_arr,
                vals_arr,
                reward_arr,
                dones_arr,
                batches,
            ) = self.memory.generate_batches()

            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr) - 1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr) - 1):
                    a_t += discount * (
                        reward_arr[k]
                        + self.gamma * values[k + 1] * (1 - int(dones_arr[k]))
                        - values[k]
                    )
                    discount *= self.gamma * self.gae_lambda
                advantage[t] = a_t
            advantage = T.tensor(advantage).to(self.device)

            values = T.tensor(values, dtype=T.float32).to(self.device)
            for batch in batches:
                states = T.tensor(state_arr[batch], dtype=T.float32).to(self.device)
                old_probs = T.tensor(old_prob_arr[batch], dtype=T.float32).to(
                    self.device
                )
                actions = T.tensor(action_arr[batch], dtype=T.float32).to(self.device)

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

                returns = advantage[batch] + values[batch]
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

        self.memory.clear_memory()
