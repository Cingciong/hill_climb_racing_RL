import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import copy
import numpy as np
import time

from sympy.abc import epsilon
from tqdm import tqdm
import matplotlib.pyplot as plt

class Agent(nn.Module):

    def __init__(self, n_actions=3, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        super(Agent, self).__init__()

        self.n_actions = n_actions
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc_image = nn.Linear(64 * 30 * 50, 48)
        self.fc_inputs = nn.Linear(3, 16)
        self.final_fc = nn.Linear(48 + 16, self.n_actions)

    def forward(self, image_input, scalar_inputs):
        x_image = F.relu(self.conv1(image_input))
        x_image = F.relu(self.conv2(x_image))
        x_image = x_image.view(x_image.size(0), -1)
        x_image = F.relu(self.fc_image(x_image))

        x_fc = F.relu(self.fc_inputs(scalar_inputs))
        if x_fc.dim() == 1:
            x_fc = x_fc.unsqueeze(0)
        combined = torch.cat((x_image, x_fc), dim=1)


        output = self.final_fc(combined)
        return output

    def get_action(self,  input_image ,input_scalar):
        input_image = torch.tensor(input_image, dtype=torch.float32)  # Convert to tensor
        input_image = input_image.view(1, 1, 30, 50)  # Add batch dimension and channel dimension (shape: [1, 1, H, W])
        input_scalar = torch.tensor(input_scalar, dtype=torch.float32)

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
        self.load_state_dict(torch.load(path))
        self.eval()

    def add_noise_to_model(self, noise_level=0.1):
        """Add random noise to model's parameters."""
        for param in self.parameters():
            noise = torch.randn_like(param) * noise_level
            param.data.add_(noise)

    def train(self, game_env, iterations=100, noise_level=0.1, max_time=5):
        best_reward = 10
        best_model = self
        best_iteration = 0
        rewards_history = []
        epsilon_history = []

        for i in tqdm(range(iterations), desc="Training Progress", ncols=100):
            start_time = time.time()

            noisy_model = copy.deepcopy(best_model)
            noisy_model.add_noise_to_model(noise_level)

            reward = self.evaluate_model_with_time_limit(noisy_model, game_env, max_time)

            rewards_history.append(reward)
            epsilon_history.append(self.epsilon)

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

        return best_model

    def plot_var(self, rewards_history,epsilon_history):
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
        plt.title("Training Progress: Rewards per Iteration")
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