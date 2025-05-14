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
        self.model = model if model else SimpleModel(dims=(15, 50, 50, 4))
        if self.display:
            pygame.init()
            self.screen = pygame.display.set_mode((game.grid.x * game.scale, game.grid.y * game.scale))
            self.clock = pygame.time.Clock()
            self.color_snake_head = (0, 255, 0)
            self.color_food = (255, 0, 0)
        self.action_space = (Vector(0, -1), Vector(0, 1), Vector(1, 0), Vector(-1, 0))

    def __del__(self):
        if self.display:
            pygame.quit()

    def update(self) -> Vector:
        dn = self.game.snake.p.y / self.game.grid.y
        de = (self.game.grid.x - self.game.snake.p.x) / self.game.grid.x
        ds = (self.game.grid.y - self.game.snake.p.y) / self.game.grid.y
        dw = self.game.snake.p.x / self.game.grid.x
        dfx = (self.game.snake.p.x - self.game.food.p.x) / self.game.grid.x
        dfy = (self.game.snake.p.y - self.game.food.p.y) / self.game.grid.y
        s = self.game.snake.score / 10.0
        danger_up = 1 if (Vector(self.game.snake.p.x, self.game.snake.p.y - 1) in self.game.snake.body or
                          self.game.snake.p.y - 1 < 0) else 0
        danger_down = 1 if (Vector(self.game.snake.p.x, self.game.snake.p.y + 1) in self.game.snake.body or
                            self.game.snake.p.y + 1 >= self.game.grid.y) else 0
        danger_right = 1 if (Vector(self.game.snake.p.x + 1, self.game.snake.p.y) in self.game.snake.body or
                             self.game.snake.p.x + 1 >= self.game.grid.x) else 0
        danger_left = 1 if (Vector(self.game.snake.p.x - 1, self.game.snake.p.y) in self.game.snake.body or
                            self.game.snake.p.x - 1 < 0) else 0
        v_x = self.game.snake.v.x
        v_y = self.game.snake.v.y
        food_right = 1 if dfx < 0 else 0
        food_down = 1 if dfy < 0 else 0
        obs = (dn, de, ds, dw, dfx, dfy, s, danger_up, danger_down, danger_right, danger_left, v_x, v_y, food_right, food_down)
        if self.display:
            print(f"Obs: {obs}")
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