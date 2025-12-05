import gymnasium as gym
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import time

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

def reinforce_eval():
    env = gym.make('CartPole-v1', render_mode='human')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy_net = PolicyNetwork(state_dim, action_dim)
    try:
        policy_net.load_state_dict(torch.load('reinforce_cartpole.pth'))
        print("Loaded trained model.")
    except FileNotFoundError:
        print("No trained model found. Running with random weights.")

    policy_net.eval()
    
    num_episodes = 5
    for i in range(num_episodes):
        state, _ = env.reset()
        done = False
        score = 0
        while not done:
            action, _ = policy_net.act(state)
            state, reward, terminated, truncated, _ = env.step(action)
            score += reward
            done = terminated or truncated
            time.sleep(0.01)
        print(f"Episode {i+1}: Score = {score}")

    env.close()

if __name__ == '__main__':
    reinforce_eval()
