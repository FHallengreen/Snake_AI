from typing import Protocol
from vector import Vector
import pygame
import time


class GameController(Protocol):
    def update(self) -> Vector:
        pass


class HumanController(GameController):
    def __init__(self, game):
        self.game = game
        self.game.controller = self
        pygame.init()
        self.screen = pygame.display.set_mode((game.grid.x * game.scale, game.grid.y * game.scale))
        self.clock = pygame.time.Clock()
        self.color_snake_head = (0, 255, 0)
        self.color_food = (255, 0, 0)
        self.font = pygame.font.Font(None, 74)
        self.small_font = pygame.font.Font(None, 36)
        self.first_update = True
        self.countdown_done = False
        self.countdown_start_time = 0

    def __del__(self):
        pygame.quit()

    def update(self) -> Vector:
        # Handle countdown for first move to give player time to react
        if self.first_update:
            self.first_update = False
            self.countdown_done = False
            self.countdown_start_time = time.time()
            self.render_screen()
            return None  # Don't move yet
            
        if not self.countdown_done:
            elapsed = time.time() - self.countdown_start_time
            if elapsed < 3:  # 3 second countdown
                self.render_countdown(3 - int(elapsed))
                return None  # Don't move during countdown
            else:
                self.countdown_done = True
                # Show "GO!" message
                self.render_countdown("GO!")
                time.sleep(0.5)  # Brief pause on "GO!"
        
        next_move = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    next_move = Vector(-1, 0)
                if event.key == pygame.K_RIGHT:
                    next_move = Vector(1, 0)
                if event.key == pygame.K_UP:
                    next_move = Vector(0, -1)
                if event.key == pygame.K_DOWN:
                    next_move = Vector(0, 1)
        
        self.render_screen()
        return next_move

    def render_screen(self):
        self.screen.fill('black')
        for i, p in enumerate(self.game.snake.body):
            pygame.draw.rect(self.screen, (0, max(128, 255 - i * 12), 0), self.block(p))
        pygame.draw.rect(self.screen, self.color_food, self.block(self.game.food.p))
        # Show current score in the corner
        score_text = self.small_font.render(f"Score: {self.game.snake.score}", True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))
        pygame.display.flip()
        self.clock.tick(10)

    def render_countdown(self, count):
        self.render_screen()
        text = self.font.render(str(count), True, (255, 255, 255))
        text_rect = text.get_rect(center=(self.game.grid.x * self.game.scale // 2, 
                                         self.game.grid.y * self.game.scale // 2))
        self.screen.blit(text, text_rect)
        pygame.display.flip()
        self.clock.tick(10)

    def block(self, obj):
        return (obj.x * self.game.scale,
                obj.y * self.game.scale,
                self.game.scale,
                self.game.scale)
