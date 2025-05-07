import pygame
import pymunk
import pymunk.pygame_util
from agent import Agent  # Ensure your agent is implemented correctly.
from terrain import Terrain  # Your terrain class
from car import Car  # Your car class
from constants import *  # Your constant values (FPS, GRAVITY, etc.)
import numpy as np
import matplotlib.pyplot as plt
import torch


class Game:
    def __init__(self):
        self.FPS = FPS
        self.SCREEN_WIDTH = WIDTH
        self.SCREEN_HEIGHT = HEIGHT
        self.space = pymunk.Space()
        self.space.gravity = (0, GRAVITY)
        self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
        self.running = True
        self.debug_mode = False

        self.x = CAR_STARTING_POSITION[0]
        self.y = CAR_STARTING_POSITION[1]
        self.car = Car(self.space, self.x, self.y, self.screen, self.debug_mode)
        self.terrain = Terrain(self.space, self.screen, seed=SEED, length=LENGTH)

        self.last_distance = 0

    def reset(self):


        del self.car
        del self.terrain

        # Re-initialize the space and gravity
        self.space = pymunk.Space()
        self.space.gravity = (0, GRAVITY)

        # Re-create the terrain and car
        self.terrain = Terrain(self.space, self.screen, seed=SEED, length=LENGTH)
        self.car = Car(self.space, self.x, self.y, self.screen, self.debug_mode)

    def get_observation(self):
        grayscale_image = self.car.get_screen_image()
        grayscale_image = pygame.surfarray.make_surface(grayscale_image)
        grayscale_image = pygame.transform.scale(grayscale_image, (30, 50))
        grayscale_image = pygame.surfarray.array3d(grayscale_image)
        grayscale_image = np.mean(grayscale_image, axis=2)
        grayscale_image = grayscale_image.astype(np.float32) / 255.0
        image_flat = grayscale_image.flatten()

        speed = self.car.get_speed()
        angle = self.car.get_angle()
        angular_velocity = self.car.get_angular_velocity()

        speed_norm = np.clip(np.array(speed / 30.0, dtype=np.float32), -1.0, 1.0)
        angle_norm = np.clip(np.array(angle / 180.0, dtype=np.float32), -1.0, 1.0)
        angular_velocity_norm = np.clip(np.array(angular_velocity / 360.0, dtype=np.float32), -1.0, 1.0)

        scalars_flat = np.array([speed_norm, angle_norm, angular_velocity_norm], dtype=np.float32)

        input_image = torch.tensor(image_flat, dtype=torch.float32).view(1, 1, 30, 50).clone().detach()
        input_scalar = torch.tensor(scalars_flat, dtype=torch.float32).clone().detach()

        return  input_image,  input_scalar

    def compute_reward(self):
        reward = self.car.distance - self.last_distance
        self.last_distance = self.car.distance
        if reward < 0:
            reward *= 2
        return reward

    def draw_background(self):
        self.screen.fill(SKY_COLOR)

    def reward(self):
        reward = self.car.distance
        print(f"Reward: {reward}")
        return reward

    def step(self, action):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

        if self.car.body.position.x <= 0:
            print("Game Over: Car hit the left edge (x ≤ 0)")
            self.car.running = False
            self.running = False
        # === Apply action to car ===
        self.car.apply_action(action)

        # === Step physics and render ===
        camera_x, camera_y = self.car.update_position(self.screen)
        self.draw_options.transform = pymunk.Transform.translation(-camera_x, -camera_y)
        self.draw_background()
        self.terrain.draw_grass(camera_x, camera_y)
        self.space.step(1 / self.FPS)
        self.space.debug_draw(self.draw_options)
        self.car.draw()

        pygame.display.flip()
        self.clock.tick(self.FPS)

        image_flat,  scalars_flat = self.get_observation()
        done = not self.car.running

        return image_flat, scalars_flat , done

    # def main_loop(self):
    #     while self.running:
    #         for event in pygame.event.get():
    #             if event.type == pygame.QUIT:
    #                 self.running = False
    #
    #         if not self.car.running:
    #             self.reset_environment()
    #
    #         self.car.apply_torque(keys=pygame.key.get_pressed())
    #
    #         reward = self.car.compute_reward()
    #         print(f"Reward: {reward}")
    #
    #         camera_x, camera_y = self.car.update_position(self.screen)
    #         self.draw_options.transform = pymunk.Transform.translation(-camera_x, -camera_y)
    #         self.draw_background()
    #         self.terrain.draw_grass(camera_x, camera_y)
    #         self.space.step(1 / self.FPS)
    #         self.space.debug_draw(self.draw_options)
    #         self.car.draw()
    #
    #
    #         pygame.display.flip()
    #         self.clock.tick(self.FPS)
    #
    #     pygame.quit()


