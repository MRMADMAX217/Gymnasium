import gymnasium as gym
from treasure_maze_env import TreasureMazeEnv
import time

def test_custom_env():
    # Instantiate directly since it's not registered
    env = TreasureMazeEnv(render_mode="human", size=7)
    
    print("Resetting env...")
    obs, info = env.reset()
    print(f"Initial Obs: {obs}")
    print(f"Info: {info}")
    
    # Run a few steps
    for i in range(10):
        action = env.action_space.sample() # Random action
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"Step {i+1}: Action={action}, Obs={obs}, Reward={reward}, Term={terminated}, Info={info}")
        
        env.render()
        time.sleep(0.1)
        
        if terminated or truncated:
            print("Episode finished!")
            obs, info = env.reset()
            
    env.close()
    print("Test passed without errors.")

if __name__ == "__main__":
    test_custom_env()
