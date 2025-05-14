import random
from collections import deque
from typing import Protocol
import pygame
from vector import Vector
from game_controller import HumanController


class SnakeGame:
    def __init__(self, xsize: int=30, ysize: int=30, scale: int=15):
        self.grid = Vector(xsize, ysize)
        self.scale = scale
        self.controller = None
        self.snake = None
        self.food = None

    def reset(self):
        self.snake = Snake(game=self)
        self.food = Food(game=self)

    def run(self, model=None, max_steps=1000, display=False, record_data=False, 
             moves=None, food_positions=None, snake_positions=None, danger_states=None):
        """
        Run the snake game
        
        Args:
            model: AI model to use (if None, uses the current controller)
            max_steps: Maximum number of steps before ending the game
            display: Whether to display the game graphically
            record_data: Whether to record detailed gameplay data
            moves: List to store movement directions (if record_data is True)
            food_positions: List to store food positions (if record_data is True)
            snake_positions: List to store snake positions (if record_data is True)
            danger_states: List to store danger states (if record_data is True)
        
        Returns:
            tuple: (score, steps)
        """
        self.reset()
        if model is not None:
            from ga_controller import GAController
            self.controller = GAController(self, display=display, model=model)
        running = True
        steps = 0
        
        # Direction mapping for data recording
        dir_mapping = {
            (0, -1): 'UP',
            (0, 1): 'DOWN',
            (1, 0): 'RIGHT',
            (-1, 0): 'LEFT'
        }
        
        while running and steps < max_steps:
            next_move = self.controller.update()
            
            if next_move:
                # Record move direction if requested
                if record_data and moves is not None:
                    moves.append(dir_mapping.get((next_move.x, next_move.y), 'NONE'))
                
                self.snake.v = next_move
                
            self.snake.move()
            steps += 1
            
            # Record game state if requested
            if record_data:
                if food_positions is not None:
                    food_positions.append((self.food.p.x, self.food.p.y))
                    
                if snake_positions is not None:
                    # Record all snake body positions
                    snake_pos = [(p.x, p.y) for p in self.snake.body]
                    snake_positions.append(snake_pos)
                    
                if danger_states is not None:
                    # Record danger state (whether positions adjacent to head are dangerous)
                    head = self.snake.p
                    dangers = {
                        'up': (Vector(head.x, head.y - 1) in self.snake.body or head.y - 1 < 0),
                        'down': (Vector(head.x, head.y + 1) in self.snake.body or head.y + 1 >= self.grid.y),
                        'right': (Vector(head.x + 1, head.y) in self.snake.body or head.x + 1 >= self.grid.x),
                        'left': (Vector(head.x - 1, head.y) in self.snake.body or head.x - 1 < 0)
                    }
                    danger_states.append(dangers)
            
            # Check game end conditions
            if not self.snake.p.within(self.grid) or self.snake.cross_own_tail:
                running = False
                
            # Handle food collection
            if self.snake.p == self.food.p:
                self.snake.add_score()
                self.food = Food(game=self)
                
        if display:
            print(f"Game over! Score: {self.snake.score}, Steps: {steps}")
            
        return self.snake.score, steps


class Food:
    def __init__(self, game: 'SnakeGame'):
        self.game = game
        self.p = Vector.random_within(self.game.grid)
        while self.p in self.game.snake.body:
            self.p = Vector.random_within(self.game.grid)


class Snake:
    def __init__(self, *, game: 'SnakeGame'):
        self.game = game
        self.score = 0
        self.v = Vector(0, 0)
        self.body = deque()
        self.body.append(Vector.random_within(self.game.grid))

    def move(self):
        self.p = self.p + self.v

    @property
    def cross_own_tail(self):
        try:
            self.body.index(self.p, 1)
            return True
        except ValueError:
            return False

    @property
    def p(self):
        return self.body[0]

    @p.setter
    def p(self, value):
        self.body.appendleft(value)
        self.body.pop()

    def add_score(self):
        self.score += 1
        tail = self.body.pop()
        self.body.append(tail)
        self.body.append(tail)

    def debug(self):
        print('===')
        for i in self.body:
            print(str(i))