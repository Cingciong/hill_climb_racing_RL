import pymunk
import pygame
from perlin_noise import PerlinNoise
from constants import *


class Terrain:

    def __init__(self, space, screen, seed=123, length=1000, step_distance=15, base_y=500, scale=0.003, amp_start=70, amp_end=300):
        self.space = space
        self.screen = screen
        self.seed = seed
        self.length = length
        self.step_distance = step_distance
        self.base_y = base_y
        self.scale = scale
        self.amp_start = amp_start
        self.amp_end = amp_end
        self.terrain_points = []
        self.generate_terrain()

    def generate_terrain(self):
        noise_gen = PerlinNoise(octaves=1, seed=self.seed)
        last_point = (0, self.base_y)

        for x in range(0, self.length, self.step_distance):
            progress = x / self.length
            amplitude = self.amp_start + (self.amp_end - self.amp_start) * progress
            n = noise_gen(x * self.scale) * 2 - 1
            y = self.base_y - n * amplitude

            self.terrain_points.append((x, y))

            segment = pymunk.Segment(self.space.static_body, last_point, (x, y), 5)
            segment.friction = 1.0
            segment.collision_type = 0
            segment.color = TERRAIN_COLOR
            self.space.add(segment)

            last_point = (x, y)

    def draw_grass(self, camera_x, camera_y):
        grass_color = (GRASS_COLOR)
        for i, (x, y) in enumerate(self.terrain_points):
            adjusted_x = x - camera_x
            adjusted_y = y - camera_y

            grass_base_y = adjusted_y - 1
            screen_bottom = self.screen.get_height()
            grass_end_y = min(screen_bottom, adjusted_y + screen_bottom)

            if 0 <= adjusted_x <= self.screen.get_width():
                pygame.draw.rect(self.screen, grass_color,
                                 (adjusted_x - self.step_distance // 2, grass_base_y, self.step_distance,
                                  grass_end_y - grass_base_y))
