import numpy as np
import gymnasium as gym
import pickle
import os
from key_door_env import KeyDoorEnv

def train():
    env = KeyDoorEnv(size=5)
    
    # Q-Learning Hyperparameters
    learning_rate = 0.1
    discount_factor = 0.99
    epsilon = 1.0
    epsilon_decay = 0.995
    min_epsilon = 0.01
    episodes = 2000

    # state_space_size = env.observation_space.n 
    # But wait, env.observation_space.n is 50.
    state_space_size = 50
    action_space_size = env.action_space.n

    q_table = np.zeros((state_space_size, action_space_size))

    for episode in range(episodes):
        state, info = env.reset()
        done = False
        total_reward = 0

        while not done:
            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[int(state), :])

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Q-Learning Update
            old_value = q_table[int(state), action]
            next_max = np.max(q_table[int(next_state), :])
            new_value = (1 - learning_rate) * old_value + learning_rate * (reward + discount_factor * next_max)
            q_table[int(state), action] = new_value

            state = next_state
            total_reward += reward

        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}: Total Reward: {total_reward}, Epsilon: {epsilon:.2f}")

    print("Training Finished.")
    
    # Save Q-table
    with open("q_table.pkl", "wb") as f:
        pickle.dump(q_table, f)
    print("Q-table saved to q_table.pkl")

if __name__ == "__main__":
    train()
