import numpy as np
import gymnasium as gym
import pickle
import time
from key_door_env import KeyDoorEnv

def eval_agent():
    # Load Q-table
    try:
        with open("q_table.pkl", "rb") as f:
            q_table = pickle.load(f)
        print("Q-table loaded successfully.")
    except FileNotFoundError:
        print("Error: q_table.pkl not found. Train the agent first.")
        return

    env = KeyDoorEnv(size=5, render_mode="human")
    
    episodes = 10
    total_steps = 0
    total_rewards = 0
    success_count = 0

    for episode in range(episodes):
        state, info = env.reset()
        done = False
        steps = 0
        episode_reward = 0
        
        print(f"Episode {episode + 1} started...")

        while not done:
            action = np.argmax(q_table[int(state), :])
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            state = next_state
            episode_reward += reward
            steps += 1
            
            # Slow down rendering for human to see
            time.sleep(0.1)

        if reward >= 9.0: # Reaching door with key gives ~9.9 (10 - 0.1)
            success_count += 1
            
        total_steps += steps
        total_rewards += episode_reward
        print(f"Episode {episode + 1} finished. Steps: {steps}, Reward: {episode_reward:.2f}")

    print("\n--- Evaluation Results ---")
    print(f"Success Rate: {success_count}/{episodes} ({success_count/episodes*100}%)")
    print(f"Average Steps: {total_steps / episodes:.2f}")
    print(f"Average Reward: {total_rewards / episodes:.2f}")
    
    env.close()

if __name__ == "__main__":
    eval_agent()
