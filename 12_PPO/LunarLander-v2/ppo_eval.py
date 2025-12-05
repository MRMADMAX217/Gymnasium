import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
import time

# Re-defining the architecture to load weights easily or importing it if better structure used
# For simplicity in this script, duplicating the class or importing would work.
# Let's import to ensure consistency if we can, but since user requested separated files before, 
# and now I put everything in train, I need to fetch it or redefine.
# I will redefine here to be self-contained as per the pattern of 'generating code' often implies standalone or clear dependencies.
# Actually, better to import to avoid code duplication if I can, but `ppo_train` runs training on import if not guarded.
# I guarded `ppo_train` with `if __name__ == '__main__':`. So I can import.

from ppo_train import ActorCritic

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def evaluate():
    env = gym.make('LunarLander-v3', render_mode='human')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    model = ActorCritic(state_dim, action_dim).to(device)
    
    try:
        model.load_state_dict(torch.load('ppo_lunarlander.pth'))
        print("Loaded trained model.")
    except FileNotFoundError:
        print("Model not found, using random weights.")
    
    model.eval()
    
    for i in range(5):
        state, _ = env.reset()
        done = False
        score = 0
        while not done:
            action, _ = model.act(state)
            state, reward, terminated, truncated, _ = env.step(action)
            score += reward
            done = terminated or truncated
            time.sleep(0.01)
        print(f"Episode {i+1}: Score = {score}")
    
    env.close()

if __name__ == '__main__':
    evaluate()
