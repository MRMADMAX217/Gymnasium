import gymnasium as gym
from gymnasium import spaces
import numpy as np

class LineWorldEnv(gym.Env):
    """
    Custom Environment: LineWorld-v0
    
    State: Integer position in {0, 1, 2, 3, 4}.
    Start: Position 0.
    Goal: Position 4. Reaching goal gives +1 and terminated=True.
    Actions: 0=Left, 1=Right.
    Per-step reward: -0.01.
    Max episode length: 20 steps (then truncated=True).
    Observation space: Discrete(5).
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None):
        self.observation_space = spaces.Discrete(5)
        self.action_space = spaces.Discrete(2)  # 0: Left, 1: Right
        self.render_mode = render_mode
        
        self._agent_location = 0
        self._step_count = 0

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        
        self._agent_location = 0
        self._step_count = 0
        
        observation = self._agent_location
        info = {}
        
        if self.render_mode == "human":
            self.render()
            
        return observation, info

    def step(self, action):
        self._step_count += 1
        
        # 0 = left, 1 = right
        if action == 1:
            self._agent_location = min(self._agent_location + 1, 4)
        elif action == 0:
            self._agent_location = max(self._agent_location - 1, 0)
        
        # Check termination condition
        terminated = (self._agent_location == 4)
        
        # Check truncation condition (max steps 20)
        truncated = (self._step_count >= 20)
        
        # Calculate reward
        if terminated:
            reward = 1.0
        else:
            reward = -0.01
            
        observation = self._agent_location
        info = {}
        
        if self.render_mode == "human":
            self.render()
            
        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            # Simple text rendering
            # Grid: [. . . . .]
            grid = ["."] * 5
            
            # Mark Agent
            grid[self._agent_location] = "A"
            
            # Mark Goal (if agent is not there)
            if self._agent_location != 4:
                grid[4] = "G"
                
            print(f"Step {self._step_count}: {' '.join(grid)}")
        
        elif self.render_mode == "rgb_array":
            # Create a small image (Height=50, Width=250, 3 channels)
            # Each cell is 50x50 pixels
            cell_size = 50
            img = np.zeros((cell_size, cell_size * 5, 3), dtype=np.uint8)
            img.fill(255) # White background
            
            # Draw Goal at index 4 (Green)
            goal_color = [100, 255, 100]
            img[:, 4*cell_size : 5*cell_size] = goal_color
            
            # Draw Agent (Blue)
            agent_color = [50, 50, 255]
            start_x = self._agent_location * cell_size
            end_x = start_x + cell_size
            img[:, start_x:end_x] = agent_color
            
            return img

# Simple usage example
if __name__ == "__main__":
    env = LineWorldEnv(render_mode="human")
    obs, info = env.reset(seed=42)
    print("Start state:", obs)
    
    terminated = False
    truncated = False
    while not (terminated or truncated):
        # Sample random action
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Action: {action}, Obs: {obs}, Reward: {reward}, Termi: {terminated}, Trunc: {truncated}")
    
    env.close()
