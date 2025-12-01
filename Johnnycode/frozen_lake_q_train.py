import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

def run_training(episodes, render=False):

    env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=True, render_mode='human' if render else None)

    q = np.zeros((env.observation_space.n, env.action_space.n)) # init a 16 x 4 array

    learning_rate_a = 0.9 # alpha or learning rate
    discount_factor_g = 0.9 # gamma or discount rate.
    epsilon = 1         # 1 = 100% random actions
    epsilon_decay_rate = 0.00002        # epsilon decay rate.
    rng = np.random.default_rng()   # random number generator

    rewards_per_episode = np.zeros(episodes)
    wins = 0

    for i in range(episodes):
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
        
        if (i+1) % 100 == 0 or i == episodes - 1:
             print(f'Episode {i+1}/{episodes} - Win Rate: {wins/(i+1):.2%}', end='\r')

    env.close()
    
    print(f"\nTraining finished.")
    print(f"Total Wins: {wins}")
    print(f"Overall Win Rate: {wins/episodes:.2%}")

    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t-100):(t+1)])
    plt.plot(sum_rewards)
    plt.savefig('frozen_lake8x8.png') # Keeping original filename as per request context, though map is 4x4 in code

    f = open("frozen_lake8x8.pkl","wb")
    pickle.dump(q, f)
    f.close()
    print("Q-table saved to frozen_lake8x8.pkl")

if __name__ == '__main__':
    run_training(50000, render=False)
