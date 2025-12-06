import numpy as np
import pickle
import time
from custom_line_world_env import LineWorldEnv

def eval_agent():
    # Load Q-table
    try:
        with open('lineworld_q_table.pkl', 'rb') as f:
            q_table = pickle.load(f)
        print("Q-table loaded successfully.")
    except FileNotFoundError:
        print("Q-table not found. Please run training first.")
        return

    env = LineWorldEnv(render_mode="human")
    
    num_episodes = 5
    max_steps_per_episode = 20
    
    for episode in range(num_episodes):
        state, info = env.reset()
        done = False
        print(f"Episode {episode + 1} started")
        
        for step in range(max_steps_per_episode):
            # Exploitation only for evaluation
            action = np.argmax(q_table[state,:])
            
            new_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            state = new_state
            
            time.sleep(0.5) # Slow down for visualization
            
            if done:
                print(f"Episode {episode + 1} finished with reward {reward} at step {step+1}")
                if terminated:
                    print("Goal reached!")
                else:
                    print("Truncated.")
                break
                
    env.close()

if __name__ == "__main__":
    eval_agent()
