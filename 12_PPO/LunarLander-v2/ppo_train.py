import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import matplotlib.pyplot as plt

# Hyperparameters
learning_rate = 0.0003
gamma = 0.99
lmbda = 0.95
eps_clip = 0.2
K_epochs = 4
T_horizon = 500 # Reduced per-env horizon, total batch = 500 * num_envs
num_envs = 8

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        
        # Actor
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
    def forward(self):
        raise NotImplementedError
    
    def act(self, state):
        state = torch.from_numpy(state).float().to(device)
        probs = self.actor(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.detach().cpu().numpy(), dist.log_prob(action).detach()
    
    def evaluate(self, state, action):
        probs = self.actor(state)
        dist = torch.distributions.Categorical(probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_value = self.critic(state)
        return action_logprobs, state_value, dist_entropy

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def train():
    env = gym.make_vec('LunarLander-v3', num_envs=num_envs, vectorization_mode='async')
    state_dim = env.single_observation_space.shape[0]
    action_dim = env.single_action_space.n
    
    model = ActorCritic(state_dim, action_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print_running_reward = 0
    print_freq = 10
    
    i_episode = 0
    
    scores = []
    
    state, _ = env.reset()
    
    while i_episode < 1000: # Max updates
        
        states = []
        actions = []
        logprobs = []
        rewards = []
        is_terminals = []
        
        for t in range(T_horizon):
            action, logprob = model.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = np.logical_or(terminated, truncated)
            
            states.append(state)
            actions.append(action)
            logprobs.append(logprob)
            rewards.append(reward)
            is_terminals.append(done)
            
            state = next_state
        
        # Calculate approximate episode reward for logging (average over envs step-wise is noisy)
        # Better: keep track of episode rewards per env
        # For simplicity in this loop, we just take mean of rewards collected * scaler or just log total steps
        # But we want "Episode Reward". 
        # With vec envs, "Episode" concept is per-env.
        # We can track completed episodes info if available, but for now let's just use average reward per step * 300?? No.
        # Let's trust the standard moving average or just sum rewards for verification.
        # Standard PPO impls often use `RecordEpisodeStatistics` wrapper.
        # But let's stick to simple logging: average reward per batch.
        
        avg_batch_reward = np.sum(rewards) / num_envs 
        
        i_episode += 1
        
        # Update PPO
        # Flatten the batch: (T, N, D) -> (T*N, D)
        states_tensor = torch.tensor(np.array(states), dtype=torch.float32).to(device) # (T, N, D)
        states_tensor = states_tensor.view(-1, state_dim)
        
        actions_tensor = torch.tensor(np.array(actions), dtype=torch.float32).to(device) # (T, N)
        actions_tensor = actions_tensor.view(-1)
        
        old_logprobs_tensor = torch.stack(logprobs).to(device).detach() # (T, N)
        old_logprobs_tensor = old_logprobs_tensor.view(-1)
        
        # Monte Carlo estimate of returns - Vectorized
        rewards_list = []
        discounted_reward = np.zeros(num_envs)
        
        for reward, done in zip(reversed(rewards), reversed(is_terminals)):
            # If done, reset gae/return
            # shape of reward: (N,), done: (N,)
            discounted_reward = reward + (gamma * discounted_reward * (1 - done))
            rewards_list.insert(0, discounted_reward)
            
        # Flatten rewards
        rewards_tensor = torch.tensor(np.array(rewards_list), dtype=torch.float32).to(device) # (T, N)
        rewards_tensor = rewards_tensor.view(-1)
        
        # Normalizing the rewards
        rewards_tensor = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-7)
        
        # Optimize policy for K epochs
        for _ in range(K_epochs):
            # Evaluating old actions and values
            # states_tensor is (T*N, D), actions_tensor is (T*N)
            logprobs, state_values, dist_entropy = model.evaluate(states_tensor, actions_tensor)
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs_tensor)
            
            # Finding Surrogate Loss
            advantages = rewards_tensor - state_values.detach()   
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-eps_clip, 1+eps_clip) * advantages
            
            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * nn.MSELoss()(state_values, rewards_tensor) - 0.01 * dist_entropy
            
            # take gradient step
            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()
            
        print_running_reward += avg_batch_reward
        scores.append(avg_batch_reward)

        if i_episode % print_freq == 0:
            avg_reward = print_running_reward / print_freq
            print(f'Update {i_episode} \t Average Batch Reward: {avg_reward:.2f}')
            print_running_reward = 0
            
    torch.save(model.state_dict(), 'ppo_lunarlander.pth')
    plt.plot(scores)
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.savefig("ppo_training_scores.png")
    env.close()

if __name__ == '__main__':
    train()
