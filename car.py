import pymunk
import pygame
import numpy as np
import torch
from constants import *  # Import your constants like car colors, etc.

CATEGORY_CAR = 0b001
CATEGORY_TERRAIN = 0b010
CATEGORY_PERSON = 0b100

CAR_MASK = CATEGORY_TERRAIN  # Car can collide with terrain
TERRAIN_MASK = CATEGORY_CAR | CATEGORY_TERRAIN | CATEGORY_PERSON  # Terrain can collide with everything


class Car:
    def __init__(self, space, x, y, screen, car_id, spring_length=20, spring_stiffness=300, damping=30):
        self.car_id = car_id
        self.space = space
        self.x = x
        self.y = y
        self.screen = screen
        self.spring_length = spring_length
        self.spring_stiffness = spring_stiffness
        self.damping = damping
        self.car_on_ground = False
        self.car_contacts = 0
        self.car_torque = 1000
        self.spin_torque = 10000
        self.point_of_force_application = (0, 10)
        self.running = True

        self.distance = 0
        self.coins = 0
        self.gasoline = 60
        self.score = 0
        self.start_x = x
        self.last_gasoline_time = pygame.time.get_ticks()
        self.person_offset = pymunk.Vec2d(0, -20)  # Adjust as needed
        self.person_stuck = True  # Start allowing the person to stay stuck

        self.create_car()
        self.add_collision_handlers()

    def create_car(self):
        # Create the car body and chassis
        self.body = pymunk.Body(1, 100)
        self.body.position = self.x, self.y
        self.chassis = pymunk.Poly.create_box(self.body, (60, 20))
        self.chassis.color = CAR_COLOR
        self.chassis.friction = 1.0

        # **Collision Filter: Exclude collision between this car's components and other cars**
        self.chassis.filter = pymunk.ShapeFilter(categories=CATEGORY_CAR, mask=(CATEGORY_TERRAIN | CATEGORY_PERSON | CATEGORY_CAR))  # Allow internal collision with person and wheels

        self.chassis.collision_type = CHASSIS
        self.space.add(self.body, self.chassis)
        self.start_x = self.body.position.x

        # Create wheels
        self.wheels = []
        for dx in [-CAR_WHEEL_SPAN, CAR_WHEEL_SPAN]:
            wheel_body = pymunk.Body(0.5, pymunk.moment_for_circle(0.5, 0, 15))
            wheel_body.position = self.x + dx, self.y + 40
            wheel = pymunk.Circle(wheel_body, 15)
            wheel.friction = 2.5
            wheel.color = WHEEL_COLOR

            # **Collision Filter for the wheels: Allow collision with chassis and person, but not other cars**
            wheel.filter = pymunk.ShapeFilter(categories=CATEGORY_CAR, mask=(CATEGORY_TERRAIN | CATEGORY_PERSON | CATEGORY_CAR))

            wheel.collision_type = WHEEL
            self.space.add(wheel_body, wheel)
            self.wheels.append(wheel_body)

            groove = pymunk.GrooveJoint(self.body, wheel_body, (dx, 10), (dx, 60), (0, 0))
            spring = pymunk.DampedSpring(self.body, wheel_body, (dx, 10), (0, 0),
                                         self.spring_length, self.spring_stiffness, self.damping)
            self.space.add(groove, spring)

        # Create person and attach it to the car's chassis
        person_mass = PERSON_MASS
        person_size = PERSON_SIZE
        person_moment = pymunk.moment_for_box(person_mass, person_size)
        self.person_body = pymunk.Body(person_mass, person_moment)
        self.person_body.position = self.body.position + self.person_offset

        self.person_shape = pymunk.Poly.create_box(self.person_body, person_size)
        self.person_shape.friction = 1.0
        self.person_shape.collision_type = PERSON
        self.person_shape.color = PERSON_COLOR

        # **Collision Filter: Person should collide with chassis and wheels, but not other cars**
        self.person_shape.filter = pymunk.ShapeFilter(categories=CATEGORY_PERSON, mask=(CATEGORY_TERRAIN | CATEGORY_CAR | CATEGORY_PERSON))

        self.space.add(self.person_body, self.person_shape)

        # Pin joint between person and chassis to keep the person stuck to the chassis
        self.pin_joint = pymunk.PinJoint(self.body, self.person_body, (0, 0), (0, 10))
        self.space.add(self.pin_joint)

    def add_collision_handlers(self):
        # Add your collision handlers here if needed
        # This is optional depending on the game behavior.
        pass

    def handle_collision_begin(self, arbiter, space, data):
        self.car_contacts += 1
        self.car_on_ground = True
        return True

    def handle_collision_separate(self, arbiter, space, data):
        self.car_contacts = max(0, self.car_contacts - 1)
        if self.car_contacts == 0:
            self.car_on_ground = False
        return True

    def update_person_position(self):
        # Update person position to stay with the car
        self.person_body.position = self.body.position + self.person_offset

    def apply_action(self, action):
        if self.car_on_ground:
            if action == 0:  # Accelerate forward
                self.body.apply_force_at_local_point((self.car_torque, 0), self.point_of_force_application)
            elif action == 1:  # Accelerate backward
                self.body.apply_force_at_local_point((-self.car_torque, 0), self.point_of_force_application)
        else:
            if action == 0:  # Spin forward
                self.body.torque += self.spin_torque
            elif action == 1:  # Spin backward
                self.body.torque -= self.spin_torque
        self.limit_speed()  # Ensure the car doesn't exceed the maximum speed

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

    def update_position(self, screen):
        car_pos_x = self.body.position.x
        car_pos_y = self.body.position.y
        screen_center_x = screen.get_width() // 3
        screen_center_y = screen.get_height() // 2
        return car_pos_x - screen_center_x, car_pos_y - screen_center_y

    def draw(self):
        self.distance = (self.body.position.x - self.start_x) / PX_IN_METER
        current_time = pygame.time.get_ticks()
        if current_time - self.last_gasoline_time >= 1000:
            self.gasoline = max(0, self.gasoline - 1)
            self.last_gasoline_time = current_time
        self.score = 1
        stats_text = f"Distance: {self.distance:.2f} m | Coins: {self.coins} | Gasoline: {self.gasoline}s | Score: {self.score}"



    def get_screen_image(self):
        surface = pygame.display.get_surface()
        width, height = surface.get_size()
        scaled_surface = pygame.transform.smoothscale(surface, MODEL_CONTEXT_WINDOW)
        raw_pixels = pygame.surfarray.array3d(scaled_surface)
        image = np.transpose(raw_pixels, (1, 0, 2)) / 255.0
        grayscale_image = np.mean(image, axis=2)
        return grayscale_image.astype(np.float32)

    def get_speed(self):
        return self.body.velocity[0] / PX_IN_METER

    def get_angle(self):
        return self.body.angle * 180 / np.pi

    def get_angular_velocity(self):
        return self.body.angular_velocity * 180 / np.pi

    def get_observation(self):
        grayscale_image = self.get_screen_image()
        grayscale_image = pygame.surfarray.make_surface(grayscale_image)
        grayscale_image = pygame.transform.scale(grayscale_image, (30, 50))
        grayscale_image = pygame.surfarray.array3d(grayscale_image)
        grayscale_image = np.mean(grayscale_image, axis=2)
        grayscale_image = grayscale_image.astype(np.float32) / 255.0
        image_flat = grayscale_image.flatten()

        speed = self.get_speed()
        angle = self.get_angle()
        angular_velocity = self.get_angular_velocity()

        speed_norm = np.clip(np.array(speed / 30.0, dtype=np.float32), -1.0, 1.0)
        angle_norm = np.clip(np.array(angle / 180.0, dtype=np.float32), -1.0, 1.0)
        angular_velocity_norm = np.clip(np.array(angular_velocity / 360.0, dtype=np.float32), -1.0, 1.0)

        scalars_flat = np.array([speed_norm, angle_norm, angular_velocity_norm], dtype=np.float32)

        input_image = torch.tensor(image_flat, dtype=torch.float32).view(1, 1, 30, 50).clone().detach()
        input_scalar = torch.tensor(scalars_flat, dtype=torch.float32).clone().detach()

        return input_image, input_scalar
