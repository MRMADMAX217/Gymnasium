import gymnasium as gym
import numpy as np
import pickle

def run_eval(episodes, render=True):
    # Create environment
    env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=True, render_mode='human' if render else None)

    # Load Q-table
    try:
        f = open('sarsa_q_table.pkl', 'rb')
        q = pickle.load(f)
        f.close()
        print("Q-table loaded successfully.")
    except FileNotFoundError:
        print("Error: sarsa_q_table.pkl not found. Please run sarsa_train.py first.")
        return

    wins = 0
    for i in range(episodes):
        # print(f'Episode {i+1}/{episodes}')
        state = env.reset()[0]
        terminated = False
        truncated = False

        while(not terminated and not truncated):
            # Always exploit (choose best action)
            action = np.argmax(q[state,:])
            
            new_state,reward,terminated,truncated,_ = env.step(action)
            state = new_state

        if reward == 1:
            wins += 1
            
    env.close()
    print(f"\nEvaluation finished.")
    print(f"Total Wins: {wins}")
    print(f"Win Rate: {wins/episodes:.2%}")

if __name__ == '__main__':
    run_eval(1000, render=True)
