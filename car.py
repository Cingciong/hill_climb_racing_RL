import pygame
import pymunk
from constants import *
import numpy as np

class Car:

    def __init__(self, space, x, y, screen, debug_mode, spring_length=20, spring_stiffness=1000, damping=30):
        self.space = space
        self.x = x
        self.y = y
        self.screen = screen
        self.debug_mode = debug_mode
        self.spring_length = spring_length
        self.spring_stiffness = spring_stiffness
        self.damping = damping
        self.car_on_ground = False
        self.car_contacts = 0
        self.car_torque = 1000
        self.spin_torque = 10000
        self.point_of_force_application = (0, 10)

        # Stats
        self.distance = 0
        self.coins = 0
        self.gasoline = 60
        self.score = 0
        self.start_x = x
        self.last_gasoline_time = pygame.time.get_ticks()

        # Create car
        self.create_car()

        # Add collision handlers
        self.add_collision_handlers()

    def create_car(self):
        self.body = pymunk.Body(1, 100)
        self.body.position = self.x, self.y
        self.chassis = pymunk.Poly.create_box(self.body, (60, 20))
        self.chassis.friction = 1.0
        self.chassis.collision_type = CHASSIS  # Use the global constant CHASSIS
        self.space.add(self.body, self.chassis)
        self.body.position = self.x, self.y
        self.start_x = self.body.position.x

        self.wheels = []
        for dx in [-25, 25]:
            wheel_body = pymunk.Body(0.5, pymunk.moment_for_circle(0.5, 0, 15))
            wheel_body.position = self.x + dx, self.y + 40
            wheel = pymunk.Circle(wheel_body, 15)
            wheel.friction = 2.5
            wheel.collision_type = WHEEL  # Use the global constant WHEEL
            self.space.add(wheel_body, wheel)
            self.wheels.append(wheel_body)

            groove = pymunk.GrooveJoint(self.body, wheel_body, (dx, 10), (dx, 60), (0, 0))
            self.space.add(groove)

            spring = pymunk.DampedSpring(self.body, wheel_body, (dx, 10), (0, 0),
                                         self.spring_length, self.spring_stiffness, self.damping)
            self.space.add(spring)

        person_mass = 0.0000001
        person_size = (25, 20)
        person_moment = pymunk.moment_for_box(person_mass, person_size)
        self.person_body = pymunk.Body(person_mass, person_moment)
        self.person_body.position = self.body.position + (0, 0)

        self.person_shape = pymunk.Poly.create_box(self.person_body, person_size)
        self.person_shape.friction = 1.0
        self.person_shape.collision_type = PERSON  # Use the global constant PERSON
        self.person_shape.color = (255, 69, 0, 255)

        self.space.add(self.person_body, self.person_shape)

        joint = pymunk.PinJoint(self.body, self.person_body, (0, 0), (0, 10))
        self.space.add(joint)

    def add_collision_handlers(self):
        handler_car_terrain = self.space.add_collision_handler(WHEEL, TERRAIN)
        handler_car_terrain.begin = self.handle_collision_begin
        handler_car_terrain.separate = self.handle_collision_separate

        handler_car_terrain_chassis = self.space.add_collision_handler(CHASSIS, TERRAIN)
        handler_car_terrain_chassis.begin = self.handle_collision_begin
        handler_car_terrain_chassis.separate = self.handle_collision_separate

        handler_person_terrain = self.space.add_collision_handler(PERSON, TERRAIN)
        handler_person_terrain.begin = self.handle_person_terrain_collision

    def handle_person_terrain_collision(self, arbiter, space, data):
        print("Person collided with the terrain. Game Over!")
        self.running = False
        return True  # Allow the collision to happen

    def handle_collision_begin(self, arbiter, space, data):
        # Check if either the wheel or the chassis is colliding with the ground (terrain)
        shapes = arbiter.shapes
        for shape in shapes:
            if shape.collision_type == WHEEL or shape.collision_type == CHASSIS:
                self.car_contacts += 1
                self.car_on_ground = True
                return True
        return False

    def handle_collision_separate(self, arbiter, space, data):
        # Check if either the wheel or the chassis has separated from the terrain
        shapes = arbiter.shapes
        for shape in shapes:
            if shape.collision_type == WHEEL or shape.collision_type == CHASSIS:
                self.car_contacts = max(0, self.car_contacts - 1)
                if self.car_contacts == 0:
                    self.car_on_ground = False
                return True
        return False

    def apply_torque(self, keys):
        if self.car_on_ground:
            if keys[pygame.K_RIGHT]:
                self.body.apply_force_at_local_point((self.car_torque, 0), self.point_of_force_application)
            if keys[pygame.K_LEFT]:
                self.body.apply_force_at_local_point((-self.car_torque, 0), self.point_of_force_application)
        else:
            if keys[pygame.K_RIGHT]:
                self.body.torque += self.spin_torque
            if keys[pygame.K_LEFT]:
                self.body.torque -= self.spin_torque

    def update_person_position(self):
        person_offset_x = 0
        person_offset_y = -30
        self.person.position = (self.body.position.x + person_offset_x, self.body.position.y + person_offset_y)

    def update_position(self, screen):
        car_pos_x = self.body.position.x
        car_pos_y = self.body.position.y
        screen_center_x = screen.get_width() // 3
        screen_center_y = screen.get_height() // 2
        camera_x = car_pos_x - screen_center_x
        camera_y = car_pos_y - screen_center_y
        return camera_x, camera_y

    def draw(self):
        self.distance = (self.body.position.x-self.start_x)/15
        current_time = pygame.time.get_ticks()
        if current_time - self.last_gasoline_time >= 1000:
            self.gasoline = max(0, self.gasoline - 1)
            self.last_gasoline_time = current_time
        self.score = 1
        stats_text = f"Distance: {self.distance} m | Coins: {self.coins} | Gasoline: {self.gasoline}s | Score: {self.score}"
        print(stats_text)

    def get_screen_image(self):
        raw_pixels = pygame.surfarray.array3d(pygame.display.get_surface())
        # Transpose to match PyTorch [C, H, W] format
        image = np.transpose(raw_pixels, (2, 0, 1))  # [3, H, W]
        image = image / 255.0  # Normalize to [0, 1]
        return image.astype(np.float32)