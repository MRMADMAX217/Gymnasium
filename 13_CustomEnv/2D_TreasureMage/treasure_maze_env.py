import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import random

class TreasureMazeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None, size=7, fixed_maze=False):
        self.size = size
        self.window_size = 512  # The size of the PyGame window
        self.fixed_maze = fixed_maze
        
        # Actions: 0=Left, 1=Right, 2=Up, 3=Down
        self.action_space = spaces.Discrete(4)
        
        # Observation: Discrete(size * size)
        self.observation_space = spaces.Discrete(size * size)
        
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        
        self.max_steps = size * size * 4
        
        self._agent_location = None
        self._treasure_location = None
        self._walls = None # Boolean mask: True = wall, False = free
        self._step_count = 0

    def _get_obs(self):
        # Return discrete index (row * size + col)
        row, col = self._agent_location
        return row * self.size + col

    def _get_info(self):
        distance = abs(self._agent_location[0] - self._treasure_location[0]) + \
                   abs(self._agent_location[1] - self._treasure_location[1])
        return {
            "distance": distance,
            "steps_taken": self._step_count
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0
        
        # Generate Maze using recursive backtracker
        # If fixed_maze is On and walls exist, skip generation
        if not (self.fixed_maze and self._walls is not None):
            self._generate_maze()
        
        # Place agents and treasure in free spots
        free_cells = []
        for r in range(self.size):
            for c in range(self.size):
                if not self._walls[r, c]:
                    free_cells.append((r, c))
        
        if len(free_cells) < 2:
            self._walls = np.zeros((self.size, self.size), dtype=bool)
            free_cells = [(r, c) for r in range(self.size) for c in range(self.size)]
            
        self._agent_location = np.array(self.np_random.choice(free_cells))
        
        # Ensure treasure is not on agent (resample agent if needed, or better: sample treasure once)
        # If fixed_maze is True and we have a treasure, keep it.
        if self.fixed_maze and self._treasure_location is not None:
             pass # Keep existing treasure
        else:
             possible_treasure = [c for c in free_cells if c != tuple(self._agent_location)]
             if not possible_treasure:
                 possible_treasure = free_cells
             self._treasure_location = np.array(self.np_random.choice(possible_treasure))
        
        # If agent spawned on fixed treasure, move agent?
        # The loop below handles it implicitly if we cared, but actually if agent spawns on treasure, 
        # the episode ends immediately. That's fine.
        
        observation = self._get_obs()
        info = self._get_info()
        
        if self.render_mode == "human":
            self._render_frame()
            
        return observation, info

    def _generate_maze(self):
        # Initialize full walls
        self._walls = np.ones((self.size, self.size), dtype=bool)
        
        # Recursive backtracker to carve paths
        # Start at (0,0) or random odd cell
        start_r, start_c = 0, 0
        self._walls[start_r, start_c] = False
        
        stack = [(start_r, start_c)]
        
        while stack:
            current_r, current_c = stack[-1]
            neighbors = []
            
            # Look for 2-step jumps to preserve walls between cells
            moves = [(-2, 0), (2, 0), (0, -2), (0, 2)]
            random.shuffle(moves) # Use random to make it interesting, though gym uses np_random usually. 
            # For pure reproducibility we should use self.np_random, but shuffle on list of tuples is easier with py random
            # Let's use self.np_random for direction selection to respect seed.
            
            # Helper to get shuffled directions using np_random
            indices = list(range(len(moves)))
            self.np_random.shuffle(indices)
            shuffled_moves = [moves[i] for i in indices]
            
            found_neighbor = False
            for dr, dc in shuffled_moves:
                nr, nc = current_r + dr, current_c + dc
                
                # Check bounds
                if 0 <= nr < self.size and 0 <= nc < self.size:
                    if self._walls[nr, nc]: # If it's a wall (unvisited)
                        # Carve path to it (remove wall at neighbor)
                        self._walls[nr, nc] = False
                        # And remove wall in between
                        between_r, between_c = current_r + dr//2, current_c + dc//2
                        self._walls[between_r, between_c] = False
                        
                        stack.append((nr, nc))
                        found_neighbor = True
                        break
            
            if not found_neighbor:
                stack.pop()
        
        # Randomly remove a few more walls to create loops/easier paths if desired?
        # User said "recursive backtracker / randomized DFS to produce perfect maze (no loops)"
        # But also "Walls force planning". Perfect maze is good.
        # However, for 7x7, sometimes 2-step jumps leave too many blocky walls if size is even/odd mismatch.
        # With size=7 (odd), 2-step jumps from (0,0) (0,2) (0,4) (0,6) works perfectly on the grid indices 0,2,4,6.
        # So we stick to strict backtracker.
        
    def step(self, action):
        # 0=Left, 1=Right, 2=Up, 3=Down
        direction = {
            0: np.array([0, -1]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([1, 0])
        }
        
        current_pos = self._agent_location
        delta = direction.get(action, np.array([0, 0]))
        new_pos = current_pos + delta
        
        # Check bounds
        if 0 <= new_pos[0] < self.size and 0 <= new_pos[1] < self.size:
            # Check walls
            if not self._walls[new_pos[0], new_pos[1]]:
                self._agent_location = new_pos
        
        self._step_count += 1
        
        terminated = np.array_equal(self._agent_location, self._treasure_location)
        truncated = self._step_count >= self.max_steps
        
        reward = 1.0 if terminated else -0.01
        
        observation = self._get_obs()
        info = self._get_info()
        
        if self.render_mode == "human":
            self._render_frame()
            
        return observation, reward, terminated, truncated, info

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
        canvas.fill((255, 255, 255)) # White floor
        
        pix_square_size = (self.window_size / self.size)
        
        # Draw walls
        for r in range(self.size):
            for c in range(self.size):
                if self._walls[r, c]:
                    pygame.draw.rect(
                        canvas,
                        (0, 0, 0), # Black walls
                        pygame.Rect(
                            pix_square_size * c,
                            pix_square_size * r,
                            pix_square_size,
                            pix_square_size,
                        ),
                    )
        
        # Draw Treasure (Gold/Yellow)
        pygame.draw.rect(
            canvas,
            (255, 215, 0),
            pygame.Rect(
                pix_square_size * self._treasure_location[1],
                pix_square_size * self._treasure_location[0],
                pix_square_size,
                pix_square_size,
            ),
        )

        # Draw Agent (Blue circle)
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (int((self._agent_location[1] + 0.5) * pix_square_size),
             int((self._agent_location[0] + 0.5) * pix_square_size)),
            int(pix_square_size / 3),
        )
        
        # Draw grid lines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=2,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=2,
            )

        if self.render_mode == "human":
            # Draw Step Count
            if not pygame.font.get_init():
                pygame.font.init()
            font = pygame.font.SysFont("Arial", 24)
            text = font.render(f"Steps: {self._step_count}", True, (0, 0, 0))
            # Draw a small background for text
            text_rect = text.get_rect()
            text_rect.topleft = (10, 10)
            pygame.draw.rect(canvas, (255, 255, 255), text_rect)
            canvas.blit(text, text_rect)

            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
