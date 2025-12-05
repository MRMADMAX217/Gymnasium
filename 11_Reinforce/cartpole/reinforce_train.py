import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)    

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.forward(state).cpu()
        m = torch.distributions.Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)

def reinforce_train():
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy_net = PolicyNetwork(state_dim, action_dim)
    optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
    gamma = 0.99
    
    num_episodes = 1000
    scores = []

    for i_episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        log_probs = []
        rewards = []
        done = False
        
        while not done:
            action, log_prob = policy_net.act(state)
            log_probs.append(log_prob)
            state, reward, terminated, truncated, _ = env.step(action)
            rewards.append(reward)
            done = terminated or truncated

        scores.append(sum(rewards))

        discounted_rewards = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            discounted_rewards.insert(0, R)
        
        discounted_rewards = torch.tensor(discounted_rewards)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)

        policy_loss = []
        for log_prob, R in zip(log_probs, discounted_rewards):
            policy_loss.append(-log_prob * R)
        
        optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        optimizer.step()

        if i_episode % 100 == 0:
            print(f'Episode {i_episode}\tAverage Score: {np.mean(scores[-100:]):.2f}')

        if np.mean(scores[-100:]) >= 195.0:
            print(f'Environment solved in {i_episode} episodes!')
            torch.save(policy_net.state_dict(), 'reinforce_cartpole.pth')
            break

    plt.plot(scores)
    plt.ylabel('Score')
    plt.xlabel('Episode')
    plt.savefig('reinforce_training_scores.png')
    env.close()

if __name__ == '__main__':
    reinforce_train()
