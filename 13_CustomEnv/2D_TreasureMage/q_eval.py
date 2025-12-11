import numpy as np
import pickle
import time
from treasure_maze_env import TreasureMazeEnv

def evaluate_agent(episodes=10, seed=42):
    env = TreasureMazeEnv(render_mode="human", size=7, fixed_maze=True)
    
    # Load Q-table
    try:
        with open("q_table.pkl", "rb") as f:
            q_table = pickle.load(f)
        print("Loaded q_table.pkl")
    except FileNotFoundError:
        print("q_table.pkl not found! Train first.")
        return

    # Initialize maze with the same seed as training
    obs, info = env.reset(seed=seed)
    
    for episode in range(episodes):
        obs, info = env.reset() # Random start, fixed maze/treasure
        done = False
        total_reward = 0
        steps = 0
        visited_counts = {}
        
        print(f"Episode {episode+1} start...")
        
        while not done:
            # Debug: Print Agent Pos and Q-values
            row, col = env._agent_location
            state_idx = row * 7 + col
            q_values = q_table[state_idx]
            
            # Update visit count
            visited_counts[state_idx] = visited_counts.get(state_idx, 0) + 1
            
            # Loop breaker: if we visited this state too many times, just randomness
            if visited_counts[state_idx] > 2:
                 # Find all valid moves to force a change
                 valid_actions = []
                 for a in range(4):
                     d_row, d_col = {0: (0,-1), 1: (0,1), 2: (-1,0), 3: (1,0)}[a]
                     nr, nc = row + d_row, col + d_col
                     if 0 <= nr < 7 and 0 <= nc < 7 and not env._walls[nr, nc]:
                         valid_actions.append(a)
                 
                 if valid_actions:
                     action = np.random.choice(valid_actions)
                     print(f"Loop detected at {state_idx} (count {visited_counts[state_idx]}), forcing valid random action: {['L','R','U','D'][action]}")
                 else:
                     action = np.random.randint(0,4) # Should not happen in maze
            else:
                # Try to find a valid move if the best one is blocked
                # Sort actions by Q-value descending
                sorted_actions = np.argsort(q_values)[::-1]
                
                for action in sorted_actions:
                    # Predict next pos
                    d_row, d_col = {0: (0,-1), 1: (0,1), 2: (-1,0), 3: (1,0)}[action]
                    nr, nc = row + d_row, col + d_col
                    
                    # Check bounds and walls
                    if 0 <= nr < 7 and 0 <= nc < 7 and not env._walls[nr, nc]:
                        # This action is valid!
                        break # Use this action
                    else:
                        # This action is blocked, skip to next best
                        continue
            
            print(f"Step {steps}: Pos=({row},{col}), Q-Vals={q_values.round(2)}, Action={['L','R','U','D'][action]}")
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
            
            
            env.render()
            # time.sleep(0.1) # Removed: env.render handles FPS
        
        status = "Success" if terminated else "Failed"
        print(f"Episode {episode+1}: {status}, Steps={steps}, Reward={total_reward:.2f}")
        time.sleep(0.5) # Pause between episodes

    print("Evaluation finished. Press Enter to exit...")
    input()
    env.close()

if __name__ == "__main__":
    try:
        evaluate_agent(episodes=50)
    except KeyboardInterrupt:
        print("\nManually interrupted.")

