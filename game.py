import pygame
import pymunk
import pymunk.pygame_util
from terrain import Terrain
from car import Car
from constants import *

class Game:
    def __init__(self):
        self.FPS = 60
        self.SCREEN_WIDTH = 1200
        self.SCREEN_HEIGHT = 800
        self.space = pymunk.Space()
        self.space.gravity = (0, 300)
        self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
        self.running = True
        self.debug_mode = False

        # Create cars dynamically
        self.cars = [Car(self.space, 200, 300, self.screen, self.debug_mode) for _ in range(1)]

        self.terrain = Terrain(self.space, self.screen, seed=123, length=20000)


    def draw_background(self):
        self.screen.fill((135, 206, 235))  # Simple sky-blue background

    def main_loop(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

            keys = pygame.key.get_pressed()
            for car in self.cars:
                car.apply_torque(keys)
                camera_x, camera_y = car.update_position(self.screen)
                self.draw_options.transform = pymunk.Transform.translation(-camera_x, -camera_y)
                self.draw_background()
                self.terrain.draw_grass(camera_x, camera_y)
                self.space.step(1 / self.FPS)
                self.space.debug_draw(self.draw_options)
                car.draw()
            pygame.display.flip()
            self.clock.tick(self.FPS)

        pygame.quit()