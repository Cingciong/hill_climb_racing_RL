import pygame
import pymunk
import pymunk.pygame_util
from perlin_noise import PerlinNoise

FPS = 60
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800

pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
clock = pygame.time.Clock()
draw_options = pymunk.pygame_util.DrawOptions(screen)

def generate_perlin_terrain(space, seed=123, length=1000, step_distance=15, base_y=500, scale=0.003, amp_start=10, amp_end=300):

    noise_gen = PerlinNoise(octaves=1, seed=seed)
    last_point = (0, base_y)

    for x in range(0, length, step_distance):
        progress = x / length
        amplitude = amp_start + (amp_end - amp_start) * progress

        # Get smooth noise value between -1 and 1
        n = noise_gen(x * scale) * 2 - 1

        y = base_y - n * amplitude

        segment = pymunk.Segment(space.static_body, last_point, (x, y), 5)
        segment.friction = 1.0
        segment.collision_type = 1
        space.add(segment)

        last_point = (x, y)

def create_car_fixed_x(x, y, spring_length=20, spring_stiffness=500, damping=30):
    # Chassis
    body = pymunk.Body(1, 100)
    body.position = x, y
    chassis = pymunk.Poly.create_box(body, (60, 20))
    chassis.friction = 1.0
    chassis.collision_type = 0
    space.add(body, chassis)

    # Wheels + vertical suspension
    wheels = []
    for dx in [-25, 25]:  # Left and right wheels
        wheel_body = pymunk.Body(0.5, pymunk.moment_for_circle(0.5, 0, 15))
        wheel_body.position = x + dx, y + 40  # ⬆️ Spawn well below the body
        wheel = pymunk.Circle(wheel_body, 15)
        wheel.friction = 2.5
        wheel.collision_type = 0
        space.add(wheel_body, wheel)
        wheels.append(wheel_body)

        # GrooveJoint to lock x position
        groove = pymunk.GrooveJoint(body, wheel_body, (dx, 10), (dx, 60), (0, 0))
        space.add(groove)

        # Suspension (adjustable parameters)
        spring = pymunk.DampedSpring(body, wheel_body, (dx, 10), (0, 0), spring_length, spring_stiffness, damping)
        space.add(spring)



    return body, wheels

def handle_collision_begin(arbiter, space, data):
    global car_on_ground, car_contacts
    car_contacts += 1
    car_on_ground = True
    return True

def handle_collision_separate(arbiter, space, data):
    global car_on_ground, car_contacts
    car_contacts = max(0, car_contacts - 1)
    if car_contacts == 0:
        car_on_ground = False
    return True

running = True
car_on_ground = False
car_contacts = 0
car_torque = 1000
spin_torque = 10000
point_of_force_application = (0, 10)
gravity = 3
seed = 123
camera_x = 0
camera_y = 0
spring_length = 20
spring_stiffness = 500
damping = 70

space = pymunk.Space()
space.gravity = (0, gravity * 100)
generate_perlin_terrain(space, seed=seed, length=20000)  # create 20,000px of terrain

left_wall = pymunk.Segment(space.static_body, (0, 0), (0, SCREEN_HEIGHT), 0)
left_wall.friction = 1.0
space.add(left_wall)

car_body, car_wheels = create_car_fixed_x(200, 300, spring_length, spring_stiffness, damping)

handler = space.add_collision_handler(0, 1)
handler.begin = handle_collision_begin
handler.separate = handle_collision_separate

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    keys = pygame.key.get_pressed()

    if car_on_ground:
        if keys[pygame.K_RIGHT]:
            car_body.apply_force_at_local_point((car_torque, 0), point_of_force_application)
        if keys[pygame.K_LEFT]:
            car_body.apply_force_at_local_point((-car_torque, 0), point_of_force_application)
    else:
        if keys[pygame.K_RIGHT]:
            car_body.torque += spin_torque  # Spin forward (clockwise)
        if keys[pygame.K_LEFT]:
            car_body.torque -= spin_torque  # Spin backward (counterclockwise)

    # Get the car's current position
    car_pos_x = car_body.position.x
    car_pos_y = car_body.position.y

    # Camera movement: center the car on the screen
    screen_center_x = screen.get_width() // 3
    screen_center_y = screen.get_height() // 2

    # Adjust the camera's x and y based on car's position
    camera_x = car_pos_x - screen_center_x
    camera_y = car_pos_y - screen_center_y

    draw_options.transform = pymunk.Transform.translation(-camera_x, -camera_y)

    # Clear screen and update space
    screen.fill((255, 255, 255))
    space.step(1 / FPS)
    space.debug_draw(draw_options)
    pygame.display.flip()
    clock.tick(FPS)
pygame.quit()
