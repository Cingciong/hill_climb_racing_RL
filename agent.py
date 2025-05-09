import random
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm


class Agent(nn.Module):
    def __init__(self, n_actions=3, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, device=None):
        super(Agent, self).__init__()

        self.n_actions = n_actions
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.device = device if device else torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')  # Default to GPU if available

        # Define a single fully connected layer for the 3 scalar inputs
        self.fc = nn.Linear(3, n_actions)  # 3 inputs -> n_actions outputs

        # Move the model to the specified device
        self.to(self.device)

    def forward(self, image_input, scalar_inputs):
        output = self.fc(scalar_inputs)
        return output

    def get_action(self, input_image, input_scalar):
        # Move inputs to device
        input_image = torch.tensor(input_image, dtype=torch.float32).to(self.device)
        input_image = input_image.view(1, 1, 30, 50)  # Add batch and channel dimensions
        input_scalar = torch.tensor(input_scalar, dtype=torch.float32).to(self.device)

        if random.random() < self.epsilon:
            action = random.choice(range(self.n_actions))
        else:
            output = self.forward(input_image, input_scalar)
            action = torch.argmax(output).item()

        return action

    def decay_epsilon(self):
        """Decay epsilon to reduce exploration over time."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load_model(self, path):
        """Load model weights from a saved file."""
        state_dict = torch.load(path, map_location=self.device)  # Ensure compatibility with device
        self.load_state_dict(state_dict)
        self.eval()

    def add_noise_to_model(self, noise_level=0.1):
        """Add random noise to model's parameters."""
        for param in self.parameters():
            noise = torch.randn_like(param) * noise_level
            param.data.add_(noise)

    def train(self, game_env, iterations=200, noise_level=0.1, max_time=30):
        best_reward = 10
        best_model = self
        best_iteration = 0
        rewards_history = []
        epsilon_history = []
        weight_history = []  # List to store weights at each iteration

        for i in tqdm(range(iterations), desc="Training Progress", ncols=100):
            start_time = time.time()

            noisy_model = copy.deepcopy(best_model)
            noisy_model.add_noise_to_model(noise_level)

            reward = self.evaluate_model_with_time_limit(noisy_model, game_env, max_time)

            rewards_history.append(reward)
            epsilon_history.append(self.epsilon)

            # Track the weights at this iteration
            weight_history.append(noisy_model.fc.weight.data.cpu().numpy())  # Store weights of fc layer

            if reward > best_reward:
                print(f"!!!!!!New best model found at iteration {i} with reward: {reward} is better than {best_reward}")

                best_reward = reward
                best_model = noisy_model
                best_iteration = i
            else:
                print(f" reward: {reward} is worse than best reward  {best_reward}")
            elapsed_time = time.time() - start_time

            if elapsed_time > max_time:
                tqdm.write(f"Iteration {i} exceeded {max_time} seconds, skipping...")
                continue
            tqdm.write(f"Iteration {i}: Current best reward: {best_reward}, Best iteration: {best_iteration}")

            self.decay_epsilon()

        # Plotting the rewards after training
        self.plot_var(rewards_history, epsilon_history)

        # Plot weights over time
        self.plot_weights_over_time(weight_history)

        return best_model

    def plot_var(self, rewards_history, epsilon_history):
        """Plot rewards history after training is completed, with linear approximation."""
        plt.figure(figsize=(10, 6))

        # Plot actual rewards
        plt.plot(rewards_history, label="Reward per iteration", color='blue')

        # Linear approximation
        x = np.arange(len(rewards_history))
        y = np.array(rewards_history)
        a, b = np.polyfit(x, y, 1)  # Fit a straight line
        y_fit = a * x + b
        plt.plot(x, y_fit, label=f"Linear Fit: y = {a:.2f}x + {b:.2f}", color='red', linestyle='--')

        plt.xlabel("Iterations")
        plt.ylabel("Reward")
        plt.title("Training Progress: Rewards per Iteration")
        plt.legend()
        plt.grid(True)
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.plot(epsilon_history, label="epsilon per iteration")
        plt.xlabel("Iterations")
        plt.ylabel("epsilon")
        plt.title("Training Progress: epsilon per Iteration")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_weights_over_time(self, weight_history):
        """Plot how the model weights change over time during training."""
        plt.figure(figsize=(10, 6))

        # Extracting the weights from weight_history to plot
        weight_history = np.array(weight_history)  # Convert list to a numpy array for easy plotting

        # Plot the weights for each neuron (this example assumes a fully connected layer)
        for i in range(weight_history.shape[1]):
            plt.plot(weight_history[:, i], label=f"Weight {i}")

        plt.xlabel("Iterations")
        plt.ylabel("Weight Value")
        plt.title("Weight Change Over Time (FC Layer)")
        plt.legend()
        plt.grid(True)
        plt.show()

    def evaluate_model_with_time_limit(self, model, game_env, max_time):
        """Evaluate the model with a time limit for the evaluation phase."""
        start_time = time.time()
        game_env.reset()
        done = False
        reward = 0
        no_action = 2  # Define a placeholder action (e.g., 'no action' or a default action)

        # Make sure to provide the 'action' parameter
        image_flat, scalars_flat, done = game_env.step(no_action)  # Providing 'no_action' here

        while not done:
            # Exit if the evaluation time exceeds the limit
            if time.time() - start_time > max_time:
                tqdm.write("Evaluation time exceeded max time, ending evaluation early.")
                break  # Stop the evaluation if it exceeds max time

            # Here, we would typically use the model to predict the next action
            action = model.get_action(image_flat, scalars_flat)

            # Pass the action as the required parameter
            image_flat, scalars_flat, done = game_env.step(action)  # Pass the action

        reward = game_env.reward()
        return reward

    def play_best_model(self, game_env, model_path='best_model.pth', max_time=30):
        """Load and play using the best pre-trained model without training."""
        self.load_model(model_path)  # Load the trained weights
        game_env.reset()
        done = False
        no_action = 2  # Default starting action
        image_flat, scalars_flat, done = game_env.step(no_action)

        start_time = time.time()

        while not done:
            if time.time() - start_time > max_time:
                print("Time limit reached.")
                break

            action = self.get_action(image_flat, scalars_flat)
            image_flat, scalars_flat, done = game_env.step(action)

        reward = game_env.reward()
        print(f"Finished episode. Final reward: {reward}")

