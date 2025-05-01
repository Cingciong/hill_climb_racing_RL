import pygame
import pymunk
import pymunk.pygame_util
from perlin_noise import PerlinNoise
import math

class Terrain:
    def __init__(self, space, screen, seed=123, length=1000, step_distance=15, base_y=500, scale=0.003, amp_start=10,
                 amp_end=300):
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
            segment.collision_type = 1
            segment.color = (172, 232, 87, 255)
            self.space.add(segment)

            last_point = (x, y)

    def draw_grass(self, camera_x, camera_y):
        grass_color = ('#695439')

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

    def plot_terrain(self):
        x_vals = [point[0] for point in self.terrain_points]
        y_vals = [point[1] for point in self.terrain_points]

        plt.figure(figsize=(12, 6))
        plt.plot(x_vals, y_vals, label='Terrain Profile', color='brown')
        plt.title('Terrain Profile using Perlin Noise')
        plt.xlabel('X Position')
        plt.ylabel('Height (Y)')
        plt.grid(True)
        plt.legend()
        plt.show()


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
        self.create_car()

    def create_car(self):
        self.car_image = pygame.image.load('car.png')
        self.car_image = pygame.transform.scale(self.car_image, (80, 25))
        self.wheel_image = pygame.image.load('wheel.png')
        self.wheel_image = pygame.transform.scale(self.wheel_image, (30, 30))

        # Create the car body (chassis)
        self.body = pymunk.Body(1, 100)
        self.body.position = self.x, self.y
        self.chassis = pymunk.Poly.create_box(self.body, (60, 20))
        self.chassis.friction = 1.0
        self.chassis.collision_type = 0
        self.chassis.color = (255, 0, 0, 255)
        self.space.add(self.body, self.chassis)

        # Create wheels with suspension
        self.wheels = []
        for dx in [-25, 25]:  # Left and right wheels
            wheel_body = pymunk.Body(0.5, pymunk.moment_for_circle(0.5, 0, 15))
            wheel_body.position = self.x + dx, self.y + 40
            wheel = pymunk.Circle(wheel_body, 15)
            wheel.friction = 2.5
            wheel.collision_type = 0
            wheel.color = (255, 255, 255, 255)
            self.space.add(wheel_body, wheel)
            self.wheels.append(wheel_body)

            # GrooveJoint to lock x position
            groove = pymunk.GrooveJoint(self.body, wheel_body, (dx, 10), (dx, 60), (0, 0))
            self.space.add(groove)

            # Suspension
            spring = pymunk.DampedSpring(self.body, wheel_body, (dx, 10), (0, 0), self.spring_length,
                                         self.spring_stiffness, self.damping)
            self.space.add(spring)

        # Create the person
        self.person = pymunk.Poly.create_box(self.body, (25, 30))  # Create a box-shaped person
        self.person.friction = 1.0
        self.person.collision_type = 2
        self.person.color = (255, 69, 0, 255)
        self.space.add(self.person)

        # Update person's position relative to the car's body (car body position + offset)
        self.update_person_position()
    def handle_collision_begin(self, arbiter, space, data):
        self.car_contacts += 1
        self.car_on_ground = True
        return True

    def handle_collision_separate(self, arbiter, space, data):
        self.car_contacts = max(0, self.car_contacts - 1)
        if self.car_contacts == 0:
            self.car_on_ground = False
        return True

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
        person_offset_x = 0  # Person should be centered relative to the car's body x position
        person_offset_y = -70  # Move the person up to be 70 pixels higher than the car body
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
        screen_center_x = self.screen.get_width() // 3
        screen_center_y = self.screen.get_height() // 2
        point_center = (screen_center_x, screen_center_y)

        rotated_car_image = pygame.transform.rotate(self.car_image, -math.degrees(self.body.angle))
        car_rect = rotated_car_image.get_rect(center=point_center)

        self.screen.blit(rotated_car_image, car_rect.topleft)

        car_pos_x = self.body.position.x
        car_pos_y = self.body.position.y
        camera_x = car_pos_x - screen_center_x
        camera_y = car_pos_y - screen_center_y

        # Update person's position based on car's body position (update every frame)
        self.update_person_position()

        # Draw the person (adjusting for camera offset)
        # Use self.person.position as a tuple directly
        person_position = (self.person.position[0] - camera_x, self.person.position[1] - camera_y)
        pygame.draw.rect(self.screen, self.person.color, pygame.Rect(person_position, (25, 30)))

        for i, wheel_body in enumerate(self.wheels):
            wheel_position = wheel_body.position
            wheel_position = (wheel_position[0] - camera_x, wheel_position[1] - camera_y)
            wheel_rotation_angle = math.degrees(wheel_body.angular_velocity)
            rotated_wheel_image = pygame.transform.rotate(self.wheel_image, wheel_rotation_angle)
            wheel_rect = rotated_wheel_image.get_rect(center=wheel_position)
            self.screen.blit(rotated_wheel_image, wheel_rect.topleft)


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
        self.car = Car(self.space, 200, 300, self.screen, self.debug_mode)
        self.terrain = Terrain(self.space, self.screen, seed=123, length=20000)

        self.background = pygame.image.load('background.png')
        self.background = pygame.transform.scale(self.background, (self.SCREEN_WIDTH, self.SCREEN_HEIGHT))

        handler_car_person = self.space.add_collision_handler(0, 2)
        handler_car_person.begin = self.handle_game_over

        handler_car_terrain = self.space.add_collision_handler(0, 1)
        handler_car_terrain.begin = self.handle_collision_begin
        handler_car_terrain.separate = self.handle_collision_separate

    def handle_game_over(self, arbiter, space, data):
        print("Game Over")
        self.running = False
        return True

    def handle_collision_begin(self, arbiter, space, data):
        self.car.car_contacts += 1
        self.car.car_on_ground = True
        return True

    def handle_collision_separate(self, arbiter, space, data):
        self.car.car_contacts = max(0, self.car.car_contacts - 1)
        if self.car.car_contacts == 0:
            self.car.car_on_ground = False
        return True

    def draw_background(self):
        self.screen.blit(self.background, (0, 0))

    def main_loop(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

            keys = pygame.key.get_pressed()

            self.car.apply_torque(keys)

            camera_x, camera_y = self.car.update_position(self.screen)

            self.draw_options.transform = pymunk.Transform.translation(-camera_x, -camera_y)

            self.draw_background()

            self.terrain.draw_grass(camera_x, camera_y)
            self.space.step(1 / self.FPS)
            self.space.debug_draw(self.draw_options)

            self.car.draw()

            pygame.display.flip()

            self.clock.tick(self.FPS)

        pygame.quit()


if __name__ == "__main__":
    game = Game()
    game.main_loop()
