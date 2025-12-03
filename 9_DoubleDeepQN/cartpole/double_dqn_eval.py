import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
import os

# Hyperparameters (must match training)
HIDDEN_SIZE = 128

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, HIDDEN_SIZE)
        self.fc2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.fc3 = nn.Linear(HIDDEN_SIZE, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

def evaluate(num_episodes=10, render=True):
    render_mode = "human" if render else None
    env = gym.make('CartPole-v1', render_mode=render_mode)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy_net = DQN(state_dim, action_dim).to(device)
    
    model_path = 'double_dqn_cartpole.pth'
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found. Train the model first.")
        return

    policy_net.load_state_dict(torch.load(model_path, map_location=device))
    policy_net.eval()

    print(f"Evaluating Double DQN on {device} for {num_episodes} episodes...")

    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        truncated = False

        while not (done or truncated):
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = policy_net(state_tensor)
                action = q_values.argmax().item()

            state, reward, done, truncated, _ = env.step(action)
            total_reward += reward

        print(f"Episode {episode+1}: Reward = {total_reward}")

    env.close()

if __name__ == "__main__":
    evaluate()
