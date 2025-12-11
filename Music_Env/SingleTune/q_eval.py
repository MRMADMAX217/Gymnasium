import numpy as np
import pickle
from env import SingleTuneEnv

# Load the Q-Table
try:
    with open("q_table.pkl", "rb") as f:
        q_table = pickle.load(f)
    print("Loaded Q-Table from 'q_table.pkl'.")
except FileNotFoundError:
    print("Error: 'q_table.pkl' not found. Please run 'python q_train.py' first.")
    exit()

# Create environment
env = SingleTuneEnv(render_mode='human', render_fps=2)

print("Evaluating Q-Learning Agent...")
obs, _ = env.reset()
state = tuple(obs.astype(int))
done = False
total_reward = 0

print(f"Goal Sequence: {env.target_sequence}\n")

step_count = 0

while not done:
    step_count += 1
    print(f"\n--- Step {step_count} ---")
    print(f"Current State: {state}")

    # Greedy action selection from Q-table
    if state in q_table:
        action = np.argmax(q_table[state])
    else:
        print(f"Warning: State {state} not found in Q-table. Choosing random action.")
        action = env.action_space.sample()
    
    print(f"Selected Action: {action}")

    obs, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    next_state = tuple(obs.astype(int))

    total_reward += reward

    # 🔥 Log reward and transition info
    print(f"Reward Received: {reward}")
    print(f"Cumulative Reward: {total_reward}")
    print(f"Next State: {next_state}")

    env.render()

    state = next_state

print("\n=== Evaluation Complete ===")
print(f"Total Reward: {total_reward}")
print(f"Final Sequence: {obs}")

# Success check
if np.array_equal(obs, env.target_sequence):
    print("SUCCESS: The Q-learning agent reproduced the target sequence correctly.")
else:
    print("FAILURE: The agent failed to reproduce the target sequence.")

env.close()
