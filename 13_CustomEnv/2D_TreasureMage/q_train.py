import numpy as np
import pickle
from treasure_maze_env import TreasureMazeEnv

def train_q_learning(episodes=10000, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.9995, min_epsilon=0.01):
    env = TreasureMazeEnv(size=7, fixed_maze=True) # Use fixed maze for tabular learning
    
    # Initialize Q-table: [States, Actions]
    q_table = np.zeros((env.observation_space.n, env.action_space.n))
    
    rewards_history = []
    
    # First reset with seed to fix the maze structure
    state, info = env.reset(seed=42)
    
    for episode in range(episodes):
        if episode > 0:
             # Subsequent resets with fixed_maze=True keep the structure but randomize agent start
             state, info = env.reset()
             
        done = False
        total_reward = 0
        
        while not done:
            # Epsilon-greedy action
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])
            
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Q-learning update
            best_next_action = np.argmax(q_table[next_state])
            td_target = reward + gamma * q_table[next_state, best_next_action] * (not done)
            td_error = td_target - q_table[state, action]
            q_table[state, action] += alpha * td_error
            
            state = next_state
            total_reward += reward
        
        # Decay epsilon
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        rewards_history.append(total_reward)
        
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode+1}/{episodes} - Reward: {total_reward:.2f} - Epsilon: {epsilon:.2f}")

    print(f"Q-Table Stats: Max={np.max(q_table):.2f}, Min={np.min(q_table):.2f}")
    # Save Q-table
    with open("q_table.pkl", "wb") as f:
        pickle.dump(q_table, f)
    print("Training finished. Q-table saved to q_table.pkl")
    
    return q_table

if __name__ == "__main__":
    train_q_learning()
