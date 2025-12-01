import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

def run_training(episodes, render=False):
    # Create environment
    env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=True, render_mode='human' if render else None)

    # Initialize Q-table
    q = np.zeros((env.observation_space.n, env.action_space.n)) # init a 16 x 4 array

    # Hyperparameters
    learning_rate_a = 0.1 # alpha or learning rate
    discount_factor_g = 0.9 # gamma or discount rate.
    epsilon = 1         # 1 = 100% random actions
    epsilon_decay_rate = 0.00005        # epsilon decay rate.
    rng = np.random.default_rng()   # random number generator

    rewards_per_episode = np.zeros(episodes)
    wins = 0

    for i in range(episodes):
        # print(f'Episode {i+1}/{episodes}', end='\r')
        state = env.reset()[0]  # states: 0 to 15
        terminated = False      # True when fall in hole or reached goal
        truncated = False       # True when actions > 200

        while(not terminated and not truncated):
            if rng.random() < epsilon:
                action = env.action_space.sample() # actions: 0=left,1=down,2=right,3=up
            else:
                action = np.argmax(q[state,:])

            new_state,reward,terminated,truncated,_ = env.step(action)

            q[state,action] = q[state,action] + learning_rate_a * (
                reward + discount_factor_g * np.max(q[new_state,:]) - q[state,action]
            )

            state = new_state

        epsilon = max(epsilon - epsilon_decay_rate, 0)

        if(epsilon==0):
            learning_rate_a = 0.001

        if reward == 1:
            rewards_per_episode[i] = 1
            wins += 1
        
        if (i + 1) % 1000 == 0:
            avg_reward = np.mean(rewards_per_episode[max(0, i-999):(i+1)])
            print(f"Episode {i+1}: Average Reward (last 1000): {avg_reward:.3f}, Win Rate (overall): {wins/(i+1):.2%}")

    env.close()

    print(f"\nTraining finished.")
    print(f"Total Episodes: {episodes}")
    print(f"Total Wins: {wins}")
    print(f"Overall Win Rate: {wins/episodes:.2%}")

    # Plot results
    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t-100):(t+1)])
    plt.plot(sum_rewards)
    plt.savefig('frozen_lake4x4_training.png')
    print("Plot saved to frozen_lake4x4_training.png")

    # Save Q-table
    f = open("frozen_lake4x4.pkl","wb")
    pickle.dump(q, f)
    f.close()
    print("Q-table saved to frozen_lake4x4.pkl")

if __name__ == '__main__':
    run_training(30000, render=False)
