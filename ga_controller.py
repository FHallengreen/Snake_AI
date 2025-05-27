from typing import Protocol
from vector import Vector
import pygame
from game_controller import GameController
from ga_models.ga_simple import SimpleModel


class GAController(GameController):
    def __init__(self, game, display=False, model=None):
        self.display = display
        self.game = game
        self.game.controller = self
        self.model = model if model else SimpleModel(dims=(21, 64, 32, 4))
        if self.display:
            pygame.init()
            self.screen = pygame.display.set_mode((game.grid.x * game.scale, game.grid.y * game.scale))
            self.clock = pygame.time.Clock()
            self.color_snake_head = (0, 255, 0)
            self.color_food = (255, 0, 0)
        self.action_space = (Vector(0, -1), Vector(0, 1), Vector(1, 0), Vector(-1, 0))
        
        # Determine input dimension from model DNA
        self.input_dim = self._detect_input_dimension()

    def _detect_input_dimension(self):
        """Detect the model's input dimension from its DNA structure"""
        if hasattr(self.model, 'DNA') and self.model.DNA:
            try:
                dna = self.model.DNA
                if isinstance(dna, list) and len(dna) > 0:
                    return dna[0].shape[0]
            except (IndexError, AttributeError):
                pass
        return 21  # Default

    def __del__(self):
        if self.display:
            pygame.quit()

    def update(self) -> Vector:
        # Normalized distances to walls (0 = close, 1 = far)
        dn = self.game.snake.p.y / self.game.grid.y
        de = (self.game.grid.x - self.game.snake.p.x) / self.game.grid.x
        ds = (self.game.grid.y - self.game.snake.p.y) / self.game.grid.y
        dw = self.game.snake.p.x / self.game.grid.x
        
        # Distance to food - normalized and signed (for direction)
        dfx = (self.game.snake.p.x - self.game.food.p.x) / self.game.grid.x
        dfy = (self.game.snake.p.y - self.game.food.p.y) / self.game.grid.y
        
        # Manhattan distance to food (normalized)
        food_dist = (abs(dfx) + abs(dfy)) / 2.0
        
        # Score normalized
        s = self.game.snake.score / 10.0
        
        # Danger detection - 1 if danger present, 0 if safe
        danger_up = 1 if (Vector(self.game.snake.p.x, self.game.snake.p.y - 1) in self.game.snake.body or
                          self.game.snake.p.y - 1 < 0) else 0
        danger_down = 1 if (Vector(self.game.snake.p.x, self.game.snake.p.y + 1) in self.game.snake.body or
                            self.game.snake.p.y + 1 >= self.game.grid.y) else 0
        danger_right = 1 if (Vector(self.game.snake.p.x + 1, self.game.snake.p.y) in self.game.snake.body or
                             self.game.snake.p.x + 1 >= self.game.grid.x) else 0
        danger_left = 1 if (Vector(self.game.snake.p.x - 1, self.game.snake.p.y) in self.game.snake.body or
                            self.game.snake.p.x - 1 < 0) else 0
                            
        # Current velocity
        v_x = self.game.snake.v.x
        v_y = self.game.snake.v.y
        
        # Direction to food (binary indicators)
        food_right = 1 if dfx < 0 else 0
        food_left = 1 if dfx > 0 else 0
        food_down = 1 if dfy < 0 else 0
        food_up = 1 if dfy > 0 else 0
        
        # Stronger food direction signals (-1 to 1 range instead of binary)
        food_dir_x = -dfx  # Negative because dfx is snake_x - food_x
        food_dir_y = -dfy  # Negative because dfy is snake_y - food_y
        
        # Body length incentive (normalized)
        body_length = len(self.game.snake.body) / (self.game.grid.x * self.game.grid.y)
        
        # Enhanced danger detection - look ahead 2 steps
        danger_up_2 = 1 if (Vector(self.game.snake.p.x, self.game.snake.p.y - 2) in self.game.snake.body or
                          self.game.snake.p.y - 2 < 0) else 0
        danger_down_2 = 1 if (Vector(self.game.snake.p.x, self.game.snake.p.y + 2) in self.game.snake.body or
                            self.game.snake.p.y + 2 >= self.game.grid.y) else 0
        danger_right_2 = 1 if (Vector(self.game.snake.p.x + 2, self.game.snake.p.y) in self.game.snake.body or
                             self.game.snake.p.x + 2 >= self.game.grid.x) else 0
        danger_left_2 = 1 if (Vector(self.game.snake.p.x - 2, self.game.snake.p.y) in self.game.snake.body or
                            self.game.snake.p.x - 2 < 0) else 0
        
        # Add diagonal danger detection 
        danger_up_right = 1 if (Vector(self.game.snake.p.x + 1, self.game.snake.p.y - 1) in self.game.snake.body or
                              self.game.snake.p.x + 1 >= self.game.grid.x or
                              self.game.snake.p.y - 1 < 0) else 0
        danger_up_left = 1 if (Vector(self.game.snake.p.x - 1, self.game.snake.p.y - 1) in self.game.snake.body or
                             self.game.snake.p.x - 1 < 0 or
                             self.game.snake.p.y - 1 < 0) else 0
        danger_down_right = 1 if (Vector(self.game.snake.p.x + 1, self.game.snake.p.y + 1) in self.game.snake.body or
                                self.game.snake.p.x + 1 >= self.game.grid.x or
                                self.game.snake.p.y + 1 >= self.game.grid.y) else 0
        danger_down_left = 1 if (Vector(self.game.snake.p.x - 1, self.game.snake.p.y + 1) in self.game.snake.body or
                               self.game.snake.p.x - 1 < 0 or
                               self.game.snake.p.y + 1 >= self.game.grid.y) else 0
        
        # Current velocity direction (one-hot encoded)
        moving_up = 1 if v_y == -1 else 0
        moving_down = 1 if v_y == 1 else 0  
        moving_right = 1 if v_x == 1 else 0
        moving_left = 1 if v_x == -1 else 0
        
        # Snake head position on grid (normalized)
        head_pos_x = self.game.snake.p.x / self.game.grid.x
        head_pos_y = self.game.snake.p.y / self.game.grid.y
        
        # Create observations based on the model's input dimension
        if self.input_dim >= 30:  # Advanced feature set
            obs = (
                # Basic wall distances (4)
                dn, de, ds, dw,
                
                # Food distances (4)
                dfx, dfy, food_dist, (abs(dfx) * abs(dfy)),  # Added food rectangle area
                
                # Snake state (2)
                s, body_length,
                
                # Immediate dangers (4)
                danger_up, danger_down, danger_right, danger_left,
                
                # Look-ahead dangers (4)
                danger_up_2, danger_down_2, danger_right_2, danger_left_2,
                
                # Diagonal dangers (4)
                danger_up_right, danger_up_left, danger_down_right, danger_down_left,
                
                # Current velocity (2)
                v_x, v_y,
                
                # Velocity one-hot encoded (4)
                moving_up, moving_down, moving_right, moving_left,
                
                # Food directions (relative) (4)
                food_up, food_down, food_right, food_left,
                
                # Food directions (continuous) (2)
                food_dir_x, food_dir_y,
                
                # Position on grid (2) - helps with spatial awareness
                head_pos_x, head_pos_y
            )
        elif self.input_dim == 21:  # Original 21-feature model
            obs = (
                dn, de, ds, dw,                               # Wall distances (4)
                dfx, dfy, food_dist,                          # Food distances (3)
                s,                                            # Score (1)
                danger_up, danger_down, danger_right, danger_left,  # Dangers (4)
                v_x, v_y,                                     # Current velocity (2)
                food_up, food_down, food_right, food_left,    # Food directions binary (4)
                food_dir_x, food_dir_y,                       # Food directions continuous (2)
                body_length                                   # Body length (1)
            )
        elif self.input_dim == 18:  # Model with 18 features
            obs = (
                dn, de, ds, dw,                               # Wall distances (4)
                dfx, dfy, food_dist,                          # Food distances (3)
                s,                                            # Score (1)
                danger_up, danger_down, danger_right, danger_left,  # Dangers (4)
                v_x, v_y,                                     # Current velocity (2)
                food_right, food_down, food_dir_x, food_dir_y # Food directions (4)
            )
        else:  # Original model format
            obs = (
                dn, de, ds, dw,                               # Wall distances (4)
                dfx, dfy,                                     # Food distances (2)
                s,                                            # Score (1)
                danger_up, danger_down, danger_right, danger_left,  # Dangers (4)
                v_x, v_y,                                     # Current velocity (2)
                food_right, food_down                         # Simple food direction (2)
            )
            
        # Calculate model output and determine next move (move outside the display condition)
        next_move = self.action_space[self.model.action(obs)]
        
        if self.display:
            self.screen.fill('black')
            for i, p in enumerate(self.game.snake.body):
                pygame.draw.rect(self.screen, (0, max(128, 255 - i * 12), 0), self.block(p))
            pygame.draw.rect(self.screen, self.color_food, self.block(self.game.food.p))
            pygame.display.flip()
            self.clock.tick(10)
        return next_move

    def block(self, obj):
        return (obj.x * self.game.scale,
                obj.y * self.game.scale,
                self.game.scale,
                self.game.scale)