import gym
import torch
import torch.nn as nn
import torch.optim as optim

from agent.net import CurvyNet


class AgentConfig:
    # Learning
    gamma = 0.99
    update_freq = 32
    k_epoch = 3
    learning_rate = 0.02
    lmbda = 0.95
    eps_clip = 0.2
    v_coef = 1
    entropy_coef = 0.01

    # Memory
    memory_size = 400

    train_cartpole = True


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Agent(AgentConfig):
    def __init__(self):
        self.env = gym.make("CartPole-v0")
        self.action_size = 2  # 2 for cartpole
        if self.train_cartpole:
            self.policy_network = CurvyNet(4, 24, self.action_size).to(device)
        self.optimizer = optim.Adam(
            self.policy_network.parameters(), lr=self.learning_rate
        )
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=self.k_epoch, gamma=0.999
        )
        self.loss = 0
        self.criterion = nn.MSELoss()
        self.memory = {
            "state": [],
            "action": [],
            "reward": [],
            "next_state": [],
            "action_prob": [],
            "terminal": [],
            "count": 0,
            "advantage": [],
            "td_target": torch.FloatTensor([]),
        }

    def new_random_game(self):
        self.env.reset()
        action = self.env.action_space.sample()
        screen, reward, terminal, info = self.env.step(action)  # type: ignore
        return screen, reward, action, terminal

    def train(self):
        episode = 0
        step = 0
        reward_history = []
        avg_reward = []
        solved = False

        # A new episode
        while not solved:
            start_step = step
            episode += 1
            episode_length = 0

            # Get initial state
            state, reward, action, terminal = self.new_random_game()
            current_state = state
            total_episode_reward = 1

            # A step in an episode
            while not solved:
                step += 1
                episode_length += 1

                # Choose action
                prob_a, _ = self.policy_network(
                    torch.FloatTensor(current_state).to(device)
                )
                # print(prob_a)
                action = torch.distributions.Categorical(prob_a).sample().item()

                # Act
                state, reward, terminal, _ = self.env.step(action)
                new_state = state

                reward = -1 if terminal else reward

                self.add_memory(
                    current_state,
                    action,
                    reward / 10.0,
                    new_state,
                    terminal,
                    prob_a[action].item(),
                )

                current_state = new_state
                total_episode_reward += reward

                if terminal:
                    episode_length = step - start_step
                    reward_history.append(total_episode_reward)
                    avg_reward.append(sum(reward_history[-10:]) / 10.0)

                    self.finish_path(episode_length)

                    if (
                        len(reward_history) > 100
                        and sum(reward_history[-100:-1]) / 100 >= 195
                    ):
                        solved = True

                    print(
                        "episode: %.2f, total step: %.2f, last_episode length: %.2f, last_episode_reward: %.2f, "
                        "loss: %.4f, lr: %.4f"
                        % (
                            episode,
                            step,
                            episode_length,
                            total_episode_reward,
                            self.loss,
                            self.scheduler.get_lr()[0],
                        )
                    )

                    self.env.reset()

                    break

            if episode % self.update_freq == 0:
                for _ in range(self.k_epoch):
                    self.update_network()

        self.env.close()

    def update_network(self):
        # get ratio
        pi, _ = self.policy_network(torch.FloatTensor(self.memory["state"]).to(device))
        new_probs_a = torch.gather(pi, 1, torch.tensor(self.memory["action"]))
        old_probs_a = torch.FloatTensor(self.memory["action_prob"])
        ratio = torch.exp(torch.log(new_probs_a) - torch.log(old_probs_a))

        # surrogate loss
        surr1 = ratio * torch.FloatTensor(self.memory["advantage"])
        surr2 = torch.clamp(
            ratio, 1 - self.eps_clip, 1 + self.eps_clip
        ) * torch.FloatTensor(self.memory["advantage"])
        _, pred_v = self.policy_network(
            torch.FloatTensor(self.memory["state"]).to(device)
        )
        v_loss = 0.5 * (pred_v - self.memory["td_target"]).pow(2)  # Huber loss
        entropy = torch.distributions.Categorical(pi).entropy()
        entropy = torch.tensor([[e] for e in entropy])
        self.loss = (
            -torch.min(surr1, surr2)
            + self.v_coef * v_loss
            - self.entropy_coef * entropy
        ).mean()

        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
        self.scheduler.step()

    def add_memory(self, s, a, r, next_s, t, prob):
        if self.memory["count"] < self.memory_size:
            self.memory["count"] += 1
        else:
            self.memory["state"] = self.memory["state"][1:]
            self.memory["action"] = self.memory["action"][1:]
            self.memory["reward"] = self.memory["reward"][1:]
            self.memory["next_state"] = self.memory["next_state"][1:]
            self.memory["terminal"] = self.memory["terminal"][1:]
            self.memory["action_prob"] = self.memory["action_prob"][1:]
            self.memory["advantage"] = self.memory["advantage"][1:]
            self.memory["td_target"] = self.memory["td_target"][1:]

        self.memory["state"].append(s)
        self.memory["action"].append([a])
        self.memory["reward"].append([r])
        self.memory["next_state"].append(next_s)
        self.memory["terminal"].append([1 - t])
        self.memory["action_prob"].append(prob)

    def finish_path(self, length):
        state = self.memory["state"][-length:]
        reward = self.memory["reward"][-length:]
        next_state = self.memory["next_state"][-length:]
        terminal = self.memory["terminal"][-length:]

        td_target = torch.FloatTensor(reward) + self.gamma * self.policy_network(
            torch.FloatTensor(next_state)
        )[1] * torch.FloatTensor(terminal)
        delta = td_target - self.policy_network(torch.FloatTensor(state))[1]
        delta = delta.detach().numpy()

        # get advantage
        advantages = []
        adv = 0.0
        for d in delta[::-1]:
            adv = self.gamma * self.lmbda * adv + d[0]
            advantages.append([adv])
        advantages.reverse()

        if self.memory["td_target"].shape == torch.Size([1, 0]):
            self.memory["td_target"] = td_target.data
        else:
            self.memory["td_target"] = torch.cat(
                (self.memory["td_target"], td_target.data), dim=0
            )
        self.memory["advantage"] += advantages


agent = Agent()
agent.train()
