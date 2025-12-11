import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

class KeyDoorEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5):
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window

        # Observations are dictionaries with the agent's and target's location.
        # However, for Q-Learning simplicity, we will return a Tuple (agent_row, agent_col, has_key)
        # mapped to a single integer or kept as a tuple if using a custom wrapper, 
        # but standard gym expectations usually want Discrete or Box.
        # To make it compatible with basic Q-Learning scripts that expect Discrete(N):
        # State = (agent_row * size + agent_col) * 2 + (1 if has_key else 0)
        # Total states = 5*5*2 = 50.
        
        self.observation_space = spaces.Discrete(size * size * 2)

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = spaces.Discrete(4)

        self._action_to_direction = {
            0: np.array([0, 1]),  # Right
            1: np.array([-1, 0]), # Up
            2: np.array([0, -1]), # Left
            3: np.array([1, 0]),  # Down
        }

        self.render_mode = render_mode
        self.window = None
        self.clock = None

        # Fixed positions for simplicity in this demo
        self._key_location = np.array([1, 1])
        self._door_location = np.array([3, 3])

    def _get_obs(self):
        # Convert state (row, col, has_key) to a single integer
        # state_idx = (row * size + col) * 2 + has_key
        row, col = self._agent_location
        has_key = 1 if self._has_key else 0
        return (row * self.size + col) * 2 + has_key

    def _get_info(self):
        return {
            "distance_to_key": np.linalg.norm(self._agent_location - self._key_location, ord=1),
            "distance_to_door": np.linalg.norm(self._agent_location - self._door_location, ord=1),
            "has_key": self._has_key
        }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)
        
        # Ensure agent doesn't spawn exactly on the key or door (optional, but good for clarity)
        # while np.array_equal(self._agent_location, self._key_location) or np.array_equal(self._agent_location, self._door_location):
        #     self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        self._has_key = False
        
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]
        
        # We use `np.clip` to make sure we don't leave the grid
        new_location = self._agent_location + direction
        self._agent_location = np.clip(new_location, 0, self.size - 1)
        
        terminated = False
        reward = -0.1 # Small step penalty
        
        # Logic:
        # 1. Check if we are at the key location and don't have it yet
        if not self._has_key and np.array_equal(self._agent_location, self._key_location):
            self._has_key = True
            reward += 1.0 # Reward for picking up key (one time)
            
        # 2. Check if we are at the door location
        if np.array_equal(self._agent_location, self._door_location):
            if self._has_key:
                reward += 10.0 # Reward for finishing
                terminated = True
            else:
                # Optional: penalty for touching door without key
                # reward -= 0.5 
                pass

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        
        pix_square_size = (self.window_size / self.size)  # The size of a single grid square in pixels

        # Draw Door (Red if locked, Green if unlocked? Let's just keep it Red)
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._door_location[::-1], # Pygame uses (col, row) aka (x, y)
                (pix_square_size, pix_square_size),
            ),
        )

        # Draw Key (Yellow) - Only if not picked up
        if not self._has_key:
            pygame.draw.rect(
                canvas,
                (255, 215, 0),
                pygame.Rect(
                    pix_square_size * self._key_location[::-1],
                    (pix_square_size, pix_square_size),
                ),
            )

        # Draw Agent (Blue)
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location[::-1] + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Draw Gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )
            
        # Draw "Key Status" text
        font = pygame.font.SysFont("Arial", 24)
        text = font.render(f"Has Key: {self._has_key}", True, (0, 0, 0))
        canvas.blit(text, (10, 10))

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
