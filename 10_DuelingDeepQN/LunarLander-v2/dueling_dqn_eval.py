import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
import time

# Hyperparameters (must match training)
HIDDEN_SIZE = 256

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DuelingDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DuelingDQN, self).__init__()
        
        # Feature layer
        self.feature_layer = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU()
        )
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, output_dim)
        )

    def forward(self, x):
        features = self.feature_layer(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        qvals = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return qvals

def evaluate():
    env = gym.make('LunarLander-v3', render_mode='human')
    
    # Get state and action space sizes
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy_net = DuelingDQN(state_dim, action_dim).to(device)
    
    try:
        policy_net.load_state_dict(torch.load('dueling_dqn_lunarlander.pth', map_location=device))
        print("Model loaded successfully.")
    except FileNotFoundError:
        print("Model file not found. Please train the agent first.")
        return

    policy_net.eval()
    
    num_episodes = 10
    rewards = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        truncated = False
        
        print(f"Starting Episode {episode + 1}")

        while not (done or truncated):
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = policy_net(state_tensor)
                action = q_values.argmax().item()

            next_state, reward, done, truncated, _ = env.step(action)
            state = next_state
            total_reward += reward
            
            # Small delay for better visualization if needed, though render_mode='human' usually handles it
            # time.sleep(0.01) 

        print(f"Episode {episode + 1} finished with reward: {total_reward:.2f}")
        rewards.append(total_reward)

    avg_reward = sum(rewards) / num_episodes
    print(f"\nAverage Reward over {num_episodes} episodes: {avg_reward:.2f}")

    env.close()

if __name__ == "__main__":
    evaluate()
