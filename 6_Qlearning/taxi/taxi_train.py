from collections import defaultdict
import gymnasium as gym
import numpy as np
import pickle
from tqdm import tqdm
from matplotlib import pyplot as plt

class TaxiAgent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ):
        """Initialize a Q-Learning agent."""
        self.env = env
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))
        self.lr = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.training_error = []

    def get_action(self, obs: int) -> int:
        """Choose an action using epsilon-greedy strategy."""
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return int(np.argmax(self.q_values[obs]))

    def update(
        self,
        obs: int,
        action: int,
        reward: float,
        terminated: bool,
        next_obs: int,
    ):
        """Update Q-value based on experience."""
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        target = reward + self.discount_factor * future_q_value
        temporal_difference = target - self.q_values[obs][action]
        self.q_values[obs][action] = (
            self.q_values[obs][action] + self.lr * temporal_difference
        )
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        """Reduce exploration rate after each episode."""
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

def get_moving_avgs(arr, window, convolution_mode):
    """Compute moving average to smooth noisy data."""
    return np.convolve(
        np.array(arr).flatten(),
        np.ones(window),
        mode=convolution_mode
    ) / window

def run_training():
    learning_rate = 0.01
    n_episodes = 20_000
    start_epsilon = 1.0
    epsilon_decay = start_epsilon / (n_episodes / 2)
    final_epsilon = 0.1

    env = gym.make("Taxi-v3")
    env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

    agent = TaxiAgent(
        env=env,
        learning_rate=learning_rate,
        initial_epsilon=start_epsilon,
        epsilon_decay=epsilon_decay,
        final_epsilon=final_epsilon,
    )

    for episode in tqdm(range(n_episodes)):
        obs, info = env.reset()
        done = False

        while not done:
            action = agent.get_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            agent.update(obs, action, reward, terminated, next_obs)
            done = terminated or truncated
            obs = next_obs

        agent.decay_epsilon()

    print(f"Initial Training Error (Avg of first 100): {np.mean(np.abs(agent.training_error[:100])):.5f}")
    print(f"Final Training Error (Avg of last 100): {np.mean(np.abs(agent.training_error[-100:])):.5f}")

    # Save Q-table
    with open("taxi_q_values.pkl", "wb") as f:
        pickle.dump(dict(agent.q_values), f)
    print("Q-table saved to taxi_q_values.pkl")

    # Plot results
    rolling_length = 500
    fig, axs = plt.subplots(ncols=3, figsize=(12, 5))

    axs[0].set_title("Episode rewards")
    reward_moving_average = get_moving_avgs(env.return_queue, rolling_length, "valid")
    axs[0].plot(range(len(reward_moving_average)), reward_moving_average)
    axs[0].set_ylabel("Average Reward")
    axs[0].set_xlabel("Episode")

    axs[1].set_title("Episode lengths")
    length_moving_average = get_moving_avgs(env.length_queue, rolling_length, "valid")
    axs[1].plot(range(len(length_moving_average)), length_moving_average)
    axs[1].set_ylabel("Average Episode Length")
    axs[1].set_xlabel("Episode")

    axs[2].set_title("Training Error")
    training_error_moving_average = get_moving_avgs(agent.training_error, rolling_length, "same")
    axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)
    axs[2].set_ylabel("Temporal Difference Error")
    axs[2].set_xlabel("Step")

    plt.tight_layout()
    plt.savefig("taxi_training_results.png")
    print("Plot saved to taxi_training_results.png")

if __name__ == "__main__":
    run_training()
