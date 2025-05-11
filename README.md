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


 






Learning breakdown
The learning process begins with the initialization, where the neural network is assigned random weights. This untrained model is then subjected to evaluation by running it in the simulated environment to measure its performance, typically in terms of distance traveled.
Following this, the process moves to mutation, where noise is deliberately added to the current best-performing model to create a slightly altered version. This new model is once again evaluated. During selection, if the mutated model shows improved performance over the current best, it replaces the best model and becomes the new reference for future mutations.
Finally, in each iteration, an epsilon adjustment is made. The exploration factor, ε, is recalculated based on how close the model is to the maximum track length. As performance improves (i.e., the model travels further), ε decreases, which results in less noise being added allowing for finer tuning in later stages of training.

r = best distance 
L = total length of track 
d = max(L - r, 0) - distance remaining 
self.epsilon = (d / L) ^ 2
So till better is mother till more fine tuning model gets i added extra ^2 for extra small noise 
Conclusions
In the training charts, we observe that the peak distance is reached around iteration 7. As a result, epsilon drops to nearly zero, which effectively disables noise injection. However, this behavior is not considered an error. Since the model is evaluated solely on the maximum distance achieved, it fulfills its objective. From this standpoint, the model performs perfectly, as it accomplishes exactly what it was trained for.
The neural network itself is intentionally simple, consisting of only 9 connections and no hidden layers. It takes in three scalar inputs: velocity, angle, and angular velocity. Which makes it susceptible to being easily misled. This simplicity, however, is by design, to accelerate training time. A typical training session lasts approximately 200 seconds.
Due to the nature of Evolutionary Algorithms (EA), it's difficult to estimate the number of epochs or iterations required to reach an optimal solution. Unlike traditional machine learning, this approach lacks gradient descent or a clear loss metric. The randomness inherent in the process makes it hard to reproduce results and find consistently strong models. The dynamic epsilon function helps address this challenge by reducing exploration as performance improves, but training larger neural networks would still require significantly more time.
Potential Improvements
·	Better Physics Engine – Using a more advanced engine like Box2D would improve simulation accuracy and make dynamics more realistic.
·	Stronger Model – Adding hidden layers could enhance learning without significantly increasing training time.
·	Image Input via CNNs – Reintroducing visual input using lightweight convolutional layers could improve decision-making based on track layout.
·	Parallel Training – Training multiple agents simultaneously would allow selection of the best-performing model and speed up optimization.
·	Improved Spring Physics – Fine-tuning suspension behavior would lead to more stable and realistic vehicle movement.
·	Faster Training – Disabling or simplifying visualization during training could greatly reduce computation time.







