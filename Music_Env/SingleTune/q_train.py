import numpy as np
import random
from env import SingleTuneEnv
import pickle

# Create environment
env = SingleTuneEnv(render_mode='human', render_fps=30)

# Q-Learning parameters
alpha = 0.1      # Learning rate
gamma = 0.99     # Discount factor
epsilon = 1.0    # Exploration rate
epsilon_decay = 0.995
min_epsilon = 0.01
episodes = 500

# Q-Table: Dictionary mapping state (tuple) to action values
# State is represented as a tuple of the sequence so far
q_table = {}

def get_q_value(state, action):
    if state not in q_table:
        q_table[state] = np.zeros(env.action_space.n)
    return q_table[state][action]

def update_q_value(state, action, reward, next_state):
    if state not in q_table:
        q_table[state] = np.zeros(env.action_space.n)
    if next_state not in q_table:
        q_table[next_state] = np.zeros(env.action_space.n)
    
    # Bellman Equation
    best_next_action = np.argmax(q_table[next_state])
    td_target = reward + gamma * q_table[next_state][best_next_action]
    td_error = td_target - q_table[state][action]
    q_table[state][action] += alpha * td_error

print("Starting Q-Learning...")

for episode in range(episodes):
    obs, info = env.reset()
    state = tuple(obs.astype(int))
    done = False
    total_reward = 0
    
    while not done:
        # Epsilon-greedy action selection
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            if state not in q_table:
                q_table[state] = np.zeros(env.action_space.n)
            action = np.argmax(q_table[state])
            
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_state = tuple(next_obs.astype(int))
        
        update_q_value(state, action, reward, next_state)
        
        state = next_state
        total_reward += reward
        
    epsilon = max(min_epsilon, epsilon * epsilon_decay)
    
    if (episode + 1) % 10 == 0:
        print(f"Episode {episode + 1}, Total Reward: {total_reward}, Epsilon: {epsilon:.3f}, Seq: {next_obs}")

# Close env after training loop
env.close()


print("Training finished.")

# Save Q-Table
with open("q_table.pkl", "wb") as f:
    pickle.dump(q_table, f)
print("Q-Table saved as 'q_table.pkl'.")

# Evaluation
print("\nEvaluating Q-Learning Agent (Internal Check):")
obs, _ = env.reset()
state = tuple(obs.astype(int))
done = False
print(f"Target: {env.target_sequence}")

path_taken = []
while not done:
    if state not in q_table:
        action = env.action_space.sample() # Should not happen if well trained
    else:
        action = np.argmax(q_table[state])
    
    obs, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    state = tuple(obs.astype(int))
    path_taken.append(action)

print(f"Agent Path: {path_taken}")
if np.array_equal(path_taken, env.target_sequence):
    print("SUCCESS: Tabular Q-Learning worked!")
else:
    print("FAILURE.")
