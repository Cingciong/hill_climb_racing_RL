Project OOP
self playing Hill Climb Racer with NN using Python 
Description of the Program
This project implements a self-playing version of the game Hill Climb Racing, controlled by a simple neural network (NN) agent. The NN is responsible for making decisions about in-game actions based on real-time input from the game environment.
Rather than using traditional reinforcement learning (RL) methods like Q-learning or policy gradients, this project adopts a simplified AE. The agent improves its performance over time by introducing random variations (noise) into its parameters and selecting the best-performing variations through repeated evaluation. Due to hardware limitations, only one strategy is evaluated per generation rather than testing a population of strategies in parallel.
Libraries used
·	matplotlib.pyplot – for ploting 
·	tqdm – For progress bars during simulations or training loops.
·	torch – PyTorch, used for building and training neural networks.
·	pygame – Game framework for rendering graphics and handling events.
·	pymunk – 2D physics engine, simulates car-terrain interaction.
·	perlinNoise – Used for generating terrain with Perlin noise

Code breakdown
Game:
·	reset - Resets the game by reinitializing the car and terrain.
·	get_observation - Retrieves and processes the current game state as an observation for the agent.
·	compute_reward - Calculates the reward based on the car’s distance.
·	draw_background - Fills the screen with the background color (sky).
·	reward - Returns the reward based on the car's distance.
·	step - Executes one step of the game, applies an action, updates the car and terrain, and returns the observation for the agent.

Terrain:
·	generate_terrain - Generates the terrain using Perlin noise and adds segments to the space.
·	draw_grass - Draws the grass on the screen based on the terrain's generated points and camera position.
Car:
·	create_car - Creates the car’s body, wheels, and joints.
·	add_collision_handlers - Adds collision handlers for the car.
·	handle_person_terrain_collision - Handles person-terrain collision (game over).
·	handle_collision_begin - Handles car collision with terrain.
·	limit_speed - Limits the car’s speed and angular velocity.
·	handle_collision_separate - Handles when the car separates from terrain.
·	apply_action - Executes agent action (turn right, left, or none).
·	update_person_position - Updates the person’s position on the car.
·	update_position - Updates camera position based on car’s position.
·	get_screen_image - Captures a scaled grayscale image of the game screen.
·	get_speed - Returns the car’s speed.
·	get_angle - Returns the car’s angle in degrees.
·	get_angular_velocity - Returns the car’s angular velocity.
·	get_observation - Returns the car’s observation (image and scalar data).

Agent:
·	forward - Defines the forward pass for the model.
·	get_action - Determines the action to take based on the inputs.
·	decay_epsilon - Decays epsilon value to reduce exploration over time.
·	load_model - Loads the model's weights from a saved file.
·	add_noise_to_model - Adds noise to the model’s parameters for exploration.
·	train - Trains the model by interacting with the environment and optimizing.
·	plot_var - Plots the training rewards and epsilon values over iterations.
·	plot_weights_over_time - Plots how the model's weights change during training.
·	evaluate_model_with_time_limit - Evaluates the model with a time limit.
·	play_best_model - Loads and plays the best model without further training.


 











