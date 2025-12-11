import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

class SingleTuneEnv(gym.Env):
    """
    Custom Environment that follows gym interface
    The goal is to reproduce a specific sequence of numbers.
    """
    metadata = {'render_modes': ['human'], 'render_fps': 4}

    def __init__(self, render_mode=None, render_fps=None):
        super(SingleTuneEnv, self).__init__()
        
        # Actions: Choose a number between 0 and 5
        self.action_space = spaces.Discrete(6)
        
        # Target sequence
        # self.target_sequence = np.array([1,2,1,5,2,3,2,1,4,5,2,1], dtype=np.int32)
        # Shorter sequence for testing/demo purposes if needed, but keeping original
        self.target_sequence = np.array([1,2,1,5,2,3,2,1,4,5,2,1,3,4], dtype=np.int32)
        self.seq_len = len(self.target_sequence)
        
        # Observation: The sequence generated so far.
        # We'll pad with -1 to indicate "empty" slots if valid numbers are 0-5.
        self.observation_space = spaces.Box(low=-1, high=5, shape=(self.seq_len,), dtype=np.int32)
        
        self.render_mode = render_mode
        # Use passed FPS or default from metadata
        if render_fps is not None:
            self.metadata['render_fps'] = render_fps
            
        self.current_step = 0
        self.current_sequence = np.full(self.seq_len, -1, dtype=np.int32)
        
        # Pygame Rendering Variables
        self.window_size = (800, 400)
        self.window = None
        self.clock = None
        self.cell_size = 50
        self.font = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.current_step = 0
        self.current_sequence = np.full(self.seq_len, -1, dtype=np.int32)
        
        if self.render_mode == "human":
            self._render_frame()
            
        return self.current_sequence, {}

    def step(self, action):
        # Update the sequence with the action taken
        if self.current_step < self.seq_len:
            self.current_sequence[self.current_step] = action
        
        # Calculate reward
        reward = 0.0
        if self.current_step < self.seq_len:
             if action == self.target_sequence[self.current_step]:
                reward = 1.0
             else:
                reward = -1.0
            
        self.current_step += 1
        
        # Check if episode is done
        terminated = self.current_step >= self.seq_len
        truncated = False
        
        if self.render_mode == "human":
            self._render_frame()
        
        return self.current_sequence, reward, terminated, truncated, {}

    def render(self):
        if self.render_mode == 'human':
            self._render_frame()

    def _render_frame(self):
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(self.window_size)
            pygame.display.set_caption("SingleTune Env")
        
        if self.clock is None:
            self.clock = pygame.time.Clock()
            
        if self.font is None:
            self.font = pygame.font.Font(None, 36)

        # Handle Pygame events (like closing the window)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return

        if self.window is None: # In case close() was called above
             return

        canvas = pygame.Surface(self.window_size)
        canvas.fill((255, 255, 255)) # White background
        
        # Draw Target Sequence
        target_text = self.font.render("Target Sequence:", True, (0, 0, 0))
        canvas.blit(target_text, (20, 50))
        
        start_x = 20
        start_y = 100
        
        for i, val in enumerate(self.target_sequence):
            color = (200, 200, 200) # Gray
            pygame.draw.rect(canvas, color, (start_x + i * (self.cell_size + 10), start_y, self.cell_size, self.cell_size))
            pygame.draw.rect(canvas, (0, 0, 0), (start_x + i * (self.cell_size + 10), start_y, self.cell_size, self.cell_size), 2)
            
            text = self.font.render(str(val), True, (0, 0, 0))
            text_rect = text.get_rect(center=(start_x + i * (self.cell_size + 10) + self.cell_size // 2, start_y + self.cell_size // 2))
            canvas.blit(text, text_rect)

        # Draw Current Sequence
        current_text = self.font.render("Current Sequence:", True, (0, 0, 0))
        canvas.blit(current_text, (20, 200))
        
        start_y_current = 250
        
        for i, val in enumerate(self.current_sequence):
            if i == self.current_step: # Highlight current step being filled (or just filled)
                 color = (255, 255, 0) # Yellow highlight for active/next slot? 
                 # Actually step fills index current_step-1. 
            elif val != -1:
                # Check directly if it matches target for simple coloring
                if val == self.target_sequence[i]:
                     color = (100, 255, 100) # Green for match
                else:
                     color = (255, 100, 100) # Red for mismatch
            else:
                color = (240, 240, 240) # Empty gray
                
            pygame.draw.rect(canvas, color, (start_x + i * (self.cell_size + 10), start_y_current, self.cell_size, self.cell_size))
            pygame.draw.rect(canvas, (0, 0, 0), (start_x + i * (self.cell_size + 10), start_y_current, self.cell_size, self.cell_size), 2)
            
            if val != -1:
                text = self.font.render(str(val), True, (0, 0, 0))
                text_rect = text.get_rect(center=(start_x + i * (self.cell_size + 10) + self.cell_size // 2, start_y_current + self.cell_size // 2))
                canvas.blit(text, text_rect)

        # Draw Step Info
        step_text = self.font.render(f"Step: {self.current_step}/{self.seq_len}", True, (0, 0, 0))
        canvas.blit(step_text, (20, 350))

        self.window.blit(canvas, (0, 0))
        pygame.display.flip()
        
        # Enforce FPS
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None
            self.font = None

