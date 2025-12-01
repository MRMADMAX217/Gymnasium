import gymnasium as gym
import numpy as np
import pickle
from collections import defaultdict

class TaxiAgent:
    def __init__(self, env, q_values=None):
        self.env = env
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))
        if q_values:
            self.q_values.update(q_values)
        self.epsilon = 0.0 # Pure exploitation for evaluation

    def get_action(self, obs: int) -> int:
        return int(np.argmax(self.q_values[obs]))

def run_eval(num_episodes=2000):
    env = gym.make("Taxi-v3")
    
    try:
        with open("taxi_q_values.pkl", "rb") as f:
            q_values = pickle.load(f)
        print("Q-table loaded successfully.")
    except FileNotFoundError:
        print("Error: taxi_q_values.pkl not found. Please run taxi_train.py first.")
        return

    agent = TaxiAgent(env, q_values)
    total_rewards = []

    for _ in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action = agent.get_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated

        total_rewards.append(episode_reward)

    # Success is defined as getting a positive reward (meaning we delivered the passenger)
    # Ideally, for "solved", we want an average reward >= 9.7
    success_rate = np.mean(np.array(total_rewards) > 0)
    average_reward = np.mean(total_rewards)

    print(f"Test Results over {num_episodes} episodes:")
    print(f"Average Reward: {average_reward:.3f} (Benchmark for 'Solved': >= 9.7)")
    print(f"Success Rate (Positive Reward): {success_rate:.1%}")
    print(f"Standard Deviation: {np.std(total_rewards):.3f}")
    

if __name__ == "__main__":
    run_eval()
