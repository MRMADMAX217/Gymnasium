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
T_horizon = 2048

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
        return action.item(), dist.log_prob(action)
    
    def evaluate(self, state, action):
        probs = self.actor(state)
        dist = torch.distributions.Categorical(probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_value = self.critic(state)
        return action_logprobs, state_value, dist_entropy

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def train():
    env = gym.make('LunarLander-v3')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    model = ActorCritic(state_dim, action_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print_running_reward = 0
    print_freq = 10
    
    time_step = 0
    i_episode = 0
    
    scores = []
    
    while i_episode < 1000: # Max episodes
        state, _ = env.reset()
        current_ep_reward = 0
        
        states = []
        actions = []
        logprobs = []
        rewards = []
        is_terminals = []
        
        for t in range(T_horizon):
            action, logprob = model.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            states.append(state)
            actions.append(action)
            logprobs.append(logprob)
            rewards.append(reward)
            is_terminals.append(done)
            
            state = next_state
            current_ep_reward += reward
            time_step += 1
            
            if done:
                break
        
        i_episode += 1
        scores.append(current_ep_reward)
        print_running_reward += current_ep_reward
        
        # Update PPO
        states_tensor = torch.tensor(np.array(states), dtype=torch.float32).to(device)
        actions_tensor = torch.tensor(np.array(actions), dtype=torch.float32).to(device)
        old_logprobs_tensor = torch.stack(logprobs).to(device).detach()
        
        # Monte Carlo estimate of returns
        rewards_list = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(rewards), reversed(is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (gamma * discounted_reward)
            rewards_list.insert(0, discounted_reward)
            
        # Normalizing the rewards
        rewards_tensor = torch.tensor(rewards_list, dtype=torch.float32).to(device)
        rewards_tensor = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-7)
        
        # Optimize policy for K epochs
        for _ in range(K_epochs):
            # Evaluating old actions and values
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
            
        if i_episode % print_freq == 0:
            avg_reward = print_running_reward / print_freq
            print(f'Episode {i_episode} \t Average Reward: {avg_reward}')
            print_running_reward = 0
            
        if np.mean(scores[-100:]) >= 200:
            print(f"Solved in {i_episode} episodes!")
            torch.save(model.state_dict(), 'ppo_lunarlander.pth')
            break
            
    plt.plot(scores)
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.savefig("ppo_training_scores.png")
    env.close()

if __name__ == '__main__':
    train()
