import numpy as np
import torch


class Memory:
    def __init__(self, device: torch.device):
        self.device = device
        self.clear()

    def remember(self, state, action, probs, values, reward: float, done: bool):
        self.states.append(state)
        self.actions.append(action.unsqueeze(1).to(self.device))
        self.values.append(values)
        self.log_probs.append(probs.unsqueeze(1).to(self.device))
        self.rewards.append(torch.FloatTensor([reward]).unsqueeze(1).to(self.device))
        self.masks.append(torch.FloatTensor([1 - done]).unsqueeze(1).to(self.device))

    def clear(self):
        self.states = []
        self.actions = []
        self.values = []
        self.log_probs = []
        self.rewards = []
        self.masks = []

    def compute_gae(self, next_value: float, gamma: float, tau: float):
        """Compute the generalized advantage estimator. The generalized advantage
        estimator is an estimate of the advantage of state s_t, which suggests how
        much better s_t is than the previous state s_t-1.

        It uses a calculation of discounted future rewards, where the discounted
        future rewards are calculated as:
            discounted_future_rewards = reward + gamma * next_value
        The advantage is then calculated as:
            advantage = discounted_future_rewards - value
        """
        values = self.values + [next_value]
        gae = 0
        returns = []
        for step in reversed(range(len(self.rewards))):
            delta = (
                self.rewards[step]
                + gamma * values[step + 1] * self.masks[step]
                - values[step]
            )
            gae = delta + gamma * tau * self.masks[step] * gae
            returns.insert(0, gae + values[step])
        return returns

    def export_for_learning(self, next_value, gamma, tau):
        returns = self.compute_gae(next_value, gamma, tau)
        returns = torch.cat(returns).detach()
        values = torch.cat(self.values).detach()
        advantage = returns - values
        print(values)
        result = Batch(
            torch.cat(self.states),
            torch.cat(self.actions),
            torch.cat(self.log_probs).detach(),
            returns,
            advantage,
        )
        self.clear()
        return result


class Batch:
    def __init__(self, states, actions, log_probs, returns, advantage):
        self.states = states
        self.actions = actions
        self.log_probs = log_probs
        self.returns = returns
        self.advantage = advantage

    def generate_minibatch(self, minibatch_size):
        """Generate a minibatch for learning. Continues to yield results until all
        data is used up."""
        batch_size = self.states.size(0)
        for _ in range(max(batch_size // minibatch_size, 1)):
            rand_ids = np.random.randint(0, batch_size, min(minibatch_size, batch_size))
            yield self.states[rand_ids, :], self.actions[rand_ids, :], self.log_probs[
                rand_ids, :
            ], self.returns[rand_ids, :], self.advantage[rand_ids, :]


class Agent:
    def __init__(self, model: torch.nn.Module, device: torch.device) -> None:

        # ----------------------------------------------------------------------
        # ATARI parameters
        # ----------------------------------------------------------------------
        # self.lr = 2.5e-4  # Learning rate (Adam stepsize)
        # self.epochs = 3  # Number of epochs to train per update
        # self.minibatch_size = 32  # Number of samples per minibatch
        # self.gamma = 0.99  # Discount factor
        # self.tau = 0.95  # Generalized advantage estimation factor (GAE)
        # self.epsilon = 0.1  # Clipping parameter for PPO
        # self.c1 = 1  # C1 hyperparameter for value loss
        # self.c2 = 0.01  # C2 hyperparameter for entropy bonus

        # ----------------------------------------------------------------------
        # Cartpole parameters
        # ----------------------------------------------------------------------
        self.lr = 0.0003  # Learning rate (Adam stepsize)
        self.epsilon = 0.2
        self.c1 = 0.5
        self.c2 = 0.01
        self.epochs = 3
        self.gamma = 0.99
        self.tau = 0.95
        self.minibatch_size = 32

        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        self.memory = Memory(device)

    def remember(self, state, action, probs, values, reward, done):
        self.memory.remember(state, action, probs, values, reward, done)

    def learn(self, next_state: torch.Tensor):
        _, next_value = self.act(next_state)
        batch = self.memory.export_for_learning(next_value, self.gamma, self.tau)
        self.ppo_update(batch)

    def ppo_update(self, batch: Batch):
        for _ in range(self.epochs):
            for (
                state,
                action,
                old_log_probs,
                return_,
                advantage,
            ) in batch.generate_minibatch(self.minibatch_size):
                dist, value = self.model(state)
                entropy = dist.entropy().mean()
                new_log_probs = dist.log_prob(action)

                ratio = (new_log_probs - old_log_probs).exp()
                surr1 = ratio * advantage
                surr2 = (
                    torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon)
                    * advantage
                )

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = (return_ - value).pow(2).mean()
                print(
                    f"Actor loss: {actor_loss.item()}, Critic loss: {critic_loss.item()}"
                )
                loss = self.c1 * critic_loss + actor_loss - self.c2 * entropy

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def act(
        self, state: torch.Tensor
    ) -> tuple[torch.distributions.Categorical, torch.Tensor]:
        return self.model(state)
