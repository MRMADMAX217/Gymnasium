import numpy as np
import random
import pickle
from custom_line_world_env import LineWorldEnv

def train_agent():
    env = LineWorldEnv()
    
    # Q-Learning parameters
    action_space_size = env.action_space.n
    state_space_size = env.observation_space.n
    
    q_table = np.zeros((state_space_size, action_space_size))
    
    num_episodes = 10000
    max_steps_per_episode = 20
    
    learning_rate = 0.1
    discount_rate = 0.99
    
    exploration_rate = 1.0
    max_exploration_rate = 1.0
    min_exploration_rate = 0.01
    exploration_decay_rate = 0.01
    
    rewards_all_episodes = []
    
    print("Starting training...")
    for episode in range(num_episodes):
        state, info = env.reset()
        done = False
        rewards_current_episode = 0
        
        for step in range(max_steps_per_episode):
            # Exploration-exploitation trade-off
            exploration_rate_threshold = random.uniform(0, 1)
            if exploration_rate_threshold > exploration_rate:
                action = np.argmax(q_table[state,:]) 
            else:
                action = env.action_space.sample()
                
            new_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Update Q-table for Q(s,a)
            q_table[state, action] = q_table[state, action] * (1 - learning_rate) + \
                learning_rate * (reward + discount_rate * np.max(q_table[new_state, :]))
                
            state = new_state
            rewards_current_episode += reward
            
            if done:
                break
                
        # Exploration rate decay
        exploration_rate = min_exploration_rate + \
            (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate * episode)
            
        rewards_all_episodes.append(rewards_current_episode)
        
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}: Average Reward: {np.mean(rewards_all_episodes[-100:])}")
            
    print("Training finished.")
    print("Updated Q-table:")
    print(q_table)
    
    # Save Q-table
    with open('lineworld_q_table.pkl', 'wb') as f:
        pickle.dump(q_table, f)
    print("Q-table saved to lineworld_q_table.pkl")

if __name__ == "__main__":
    train_agent()
