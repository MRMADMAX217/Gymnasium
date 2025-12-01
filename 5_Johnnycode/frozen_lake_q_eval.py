import gymnasium as gym
import numpy as np
import pickle

def run_eval(episodes, render=True):

    env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=True, render_mode='human' if render else None)

    try:
        f = open('frozen_lake8x8.pkl', 'rb')
        q = pickle.load(f)
        f.close()
        print("Q-table loaded successfully.")
    except FileNotFoundError:
        print("Error: frozen_lake8x8.pkl not found. Please run frozen_lake_q_train.py first.")
        return

    wins = 0

    for i in range(episodes):
        state = env.reset()[0]  # states: 0 to 15
        terminated = False      # True when fall in hole or reached goal
        truncated = False       # True when actions > 200

        while(not terminated and not truncated):
            action = np.argmax(q[state,:])
            new_state,reward,terminated,truncated,_ = env.step(action)
            state = new_state

        if reward == 1:
            wins += 1
        
        if (i+1) % 100 == 0 or i == episodes - 1:
             print(f'Episode {i+1}/{episodes} - Win Rate: {wins/(i+1):.2%}', end='\r')

    env.close()
    
    print(f"\nEvaluation finished.")
    print(f"Total Wins: {wins}")
    print(f"Overall Win Rate: {wins/episodes:.2%}")

if __name__ == '__main__':
    run_eval(1000, render=False)
