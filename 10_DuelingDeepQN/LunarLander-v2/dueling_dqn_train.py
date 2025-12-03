import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
import os

# Hyperparameters
LEARNING_RATE = 0.0005
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
BATCH_SIZE = 64
MEMORY_SIZE = 100000
TARGET_UPDATE = 4  # Update every 4 episodes (soft update or hard update frequency)
NUM_EPISODES = 1000
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

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (np.array(state), np.array(action), np.array(reward), 
                np.array(next_state), np.array(done))

    def __len__(self):
        return len(self.buffer)

def train():
    env = gym.make('LunarLander-v3')
    
    # Get state and action space sizes
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy_net = DuelingDQN(state_dim, action_dim).to(device)
    target_net = DuelingDQN(state_dim, action_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    memory = ReplayBuffer(MEMORY_SIZE)

    epsilon = EPSILON_START
    episode_rewards = []
    
    print(f"Training Dueling DQN on {device}...")

    for episode in range(NUM_EPISODES):
        state, _ = env.reset()
        total_reward = 0
        done = False
        truncated = False

        while not (done or truncated):
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                    q_values = policy_net(state_tensor)
                    action = q_values.argmax().item()

            next_state, reward, done, truncated, _ = env.step(action)
            
            # Store transition
            memory.push(state, action, reward, next_state, done or truncated)
            
            state = next_state
            total_reward += reward

            # Optimize model
            if len(memory) >= BATCH_SIZE:
                states, actions, rewards, next_states, dones = memory.sample(BATCH_SIZE)

                states = torch.FloatTensor(states).to(device)
                actions = torch.LongTensor(actions).unsqueeze(1).to(device)
                rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
                next_states = torch.FloatTensor(next_states).to(device)
                dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

                # Compute Q(s, a)
                q_values = policy_net(states).gather(1, actions)

                # Double DQN Target Calculation using Dueling Network
                with torch.no_grad():
                    # 1. Select best action using policy net
                    best_actions = policy_net(next_states).argmax(1).unsqueeze(1)
                    # 2. Evaluate that action using target net
                    next_q_values = target_net(next_states).gather(1, best_actions)
                    
                    expected_q_values = rewards + (GAMMA * next_q_values * (1 - dones))

                loss = nn.MSELoss()(q_values, expected_q_values)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Update target network
        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # Decay epsilon
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
        
        episode_rewards.append(total_reward)
        
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Episode {episode+1}/{NUM_EPISODES}, Reward: {total_reward:.2f}, Avg Reward (last 10): {avg_reward:.2f}, Epsilon: {epsilon:.2f}")

    env.close()

    # Save model
    torch.save(policy_net.state_dict(), 'dueling_dqn_lunarlander.pth')
    print("Model saved to dueling_dqn_lunarlander.pth")

    # Plot results
    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards)
    plt.title('Dueling DQN Training Rewards (LunarLander-v2)')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.savefig('dueling_dqn_training.png')
    print("Training plot saved to dueling_dqn_training.png")

if __name__ == "__main__":
    train()
