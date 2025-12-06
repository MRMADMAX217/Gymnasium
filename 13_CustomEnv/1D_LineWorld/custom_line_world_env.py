import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

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

        self.window_size = 512
        self.window = None
        self.clock = None

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
        if self.render_mode == "rgb_array":
            return self._render_frame()

        elif self.render_mode == "human":
            self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, int(self.window_size / 5)))
        
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, int(self.window_size / 5)))
        canvas.fill((255, 255, 255))
        
        pix_square_size = (
            self.window_size / 5
        )  # The size of a single grid square in pixels

        # Draw goal
        pygame.draw.rect(
            canvas,
            (0, 255, 0),
            pygame.Rect(
                4 * pix_square_size, 0, pix_square_size, pix_square_size
            ),
        )

        # Draw agent
        pygame.draw.rect(
            canvas,
            (0, 0, 255),
            pygame.Rect(
                self._agent_location * pix_square_size, 0, pix_square_size, pix_square_size
            ),
        )

        # Add some gridlines
        for x in range(5 + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, 0),
                (0, self.window_size / 5),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (x * pix_square_size, 0),
                (x * pix_square_size, self.window_size / 5),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
