import pygame
import pymunk
from constants import *
import numpy as np
import matplotlib.pyplot as plt

class Car:

    def __init__(self, space, x, y, screen, debug_mode, spring_length=20, spring_stiffness=300, damping=30):
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
        self.running = True


        # Stats
        self.distance = 0
        self.coins = 0
        self.gasoline = 60
        self.score = 0
        self.start_x = x
        self.last_gasoline_time = pygame.time.get_ticks()
        self.last_distance = 0

        # Create car
        self.create_car()

        # Add collision handlers
        self.add_collision_handlers()

    def create_car(self):
        self.body = pymunk.Body(1, 100)
        self.body.position = self.x, self.y
        self.chassis = pymunk.Poly.create_box(self.body, (60, 20))
        self.chassis.color = CAR_COLOR
        self.chassis.friction = 1.0
        self.chassis.collision_type = CHASSIS  # Use the global constant CHASSIS
        self.space.add(self.body, self.chassis)
        self.body.position = self.x, self.y
        self.start_x = self.body.position.x

        self.wheels = []
        for dx in [-CAR_WHEEL_SPAN, CAR_WHEEL_SPAN]:
            wheel_body = pymunk.Body(0.5, pymunk.moment_for_circle(0.5, 0, 15))
            wheel_body.position = self.x + dx, self.y + 40
            wheel = pymunk.Circle(wheel_body, 15)
            wheel.friction = 2.5
            wheel.color = WHEEL_COLOR
            wheel.collision_type = WHEEL  # Use the global constant WHEEL
            self.space.add(wheel_body, wheel)
            self.wheels.append(wheel_body)

            groove = pymunk.GrooveJoint(self.body, wheel_body, (dx, 10), (dx, 60), (0, 0))
            self.space.add(groove)

            spring = pymunk.DampedSpring(self.body, wheel_body, (dx, 10), (0, 0),
                                         self.spring_length, self.spring_stiffness, self.damping)
            self.space.add(spring)

        person_mass = PERSON_MASS
        person_size = PERSON_SIZE
        person_moment = pymunk.moment_for_box(person_mass, person_size)
        self.person_body = pymunk.Body(person_mass, person_moment)
        self.person_body.position = self.body.position + (0, 0)

        self.person_shape = pymunk.Poly.create_box(self.person_body, person_size)
        self.person_shape.friction = 1.0
        self.person_shape.collision_type = PERSON
        self.person_shape.color = PERSON_COLOR

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
        return True

    def handle_collision_begin(self, arbiter, space, data):
        # Check if either the wheel or the chassis is colliding with the ground (terrain)
        shapes = arbiter.shapes
        for shape in shapes:
            if shape.collision_type == WHEEL or shape.collision_type == CHASSIS:
                self.car_contacts += 1
                self.car_on_ground = True
                return True
        return False

    def limit_speed(self):
        # Limit linear velocity
        max_velocity = MAX_SPEED * PX_IN_METER  # convert to pixels/second
        velocity = self.body.velocity
        speed = velocity.length
        if speed > max_velocity:
            scale = max_velocity / speed
            self.body.velocity = velocity * scale

        # Limit angular velocity (convert max to radians/sec)
        max_angular_velocity = np.radians(MAX_ANGULAR_SPEED_DEG)  # e.g., 360 deg/s
        angular_speed = abs(self.body.angular_velocity)
        if angular_speed > max_angular_velocity:
            self.body.angular_velocity = np.sign(self.body.angular_velocity) * max_angular_velocity

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

        self.limit_speed()

    def apply_action(self, action):
        """
        Apply the action chosen by the agent.
        action == 0 -> Turn right (apply positive torque)
        action == 1 -> Turn left (apply negative torque)
        action == 2 -> No action (do nothing)
        """
        if self.car_on_ground:
            if action == 0:  # Turn right (simulate turning right)
                self.body.apply_force_at_local_point((self.car_torque, 0), self.point_of_force_application)
            elif action == 1:  # Turn left (simulate turning left)
                self.body.apply_force_at_local_point((-self.car_torque, 0), self.point_of_force_application)
        else:
            if action == 0:  # Turn right (apply spin torque)
                self.body.torque += self.spin_torque
            elif action == 1:  # Turn left (apply spin torque)
                self.body.torque -= self.spin_torque

        self.limit_speed()  # Limit the car's speed if necessary

    def update_person_position(self):
        person_offset_x = 0
        person_offset_y = PERSON_OFFSET
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
        self.distance = (self.body.position.x-self.start_x)/PX_IN_METER
        current_time = pygame.time.get_ticks()
        if current_time - self.last_gasoline_time >= 1000:
            self.gasoline = max(0, self.gasoline - 1)
            self.last_gasoline_time = current_time
        self.score = 1
        stats_text = f"Distance: {self.distance} m | Coins: {self.coins} | Gasoline: {self.gasoline}s | Score: {self.score}"


    def get_screen_image(self):
        surface = pygame.display.get_surface()
        width, height = surface.get_size()

        scaled_width,scaled_height = MODEL_CONTEXT_WINDOW

        scaled_surface = pygame.transform.smoothscale(surface, (scaled_width, scaled_height))

        raw_pixels = pygame.surfarray.array3d(scaled_surface)  # [W, H, 3]
        image = np.transpose(raw_pixels, (1, 0, 2)) / 255.0  # [H, W, 3]
        grayscale_image = np.mean(image, axis=2)  # [H, W]

        return grayscale_image.astype(np.float32)

    def get_speed(self):
        return self.body.velocity[0] / PX_IN_METER

    def get_angle(self):
        return self.body.angle * 180 / np.pi

    def get_angular_velocity(self):
        return self.body.angular_velocity * 180 / np.pi

    def get_observation(self):
        grayscale_image = self.get_screen_image()  # shape: (H, W)
        image_flat = grayscale_image.flatten()  # shape: (H * W,)

        # Get and normalize scalar features
        speed = self.get_speed()  # m/s
        angle = self.get_angle()  # degrees
        angular_velocity = self.get_angular_velocity()  # deg/s

        speed_norm = np.clip(speed / 30.0, -1.0, 1.0)  # assume max speed ~30 m/s
        angle_norm = np.clip(angle / 180.0, -1.0, 1.0)  # degrees -> [-1, 1]
        angular_velocity_norm = np.clip(angular_velocity / 360.0, -1.0, 1.0)  # assume ±360 deg/s

        scalars = np.array([speed_norm, angle_norm, angular_velocity_norm], dtype=np.float32)

        return {"image": image_flat, "scalars": scalars}






