import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import copy
import numpy as np
import time
import hashlib
import seaborn as sns

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

        self.fc1 = nn.Linear(3, 8)
        self.fc2 = nn.Linear(8, n_actions)


    def forward(self, image_flat, scalars_flat):
        x = F.relu(self.fc1(scalars_flat))
        x = self.fc2(x)
        return x

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

    def model_signature(model):
        # Creates a hash based on all parameters in the model
        params = torch.cat([p.flatten().detach().cpu() for p in model.parameters()])
        return hashlib.md5(params.numpy().tobytes()).hexdigest()[:8]

    def add_noise_to_model(self, noise_level):
        for param in self.parameters():
            if param.requires_grad:
                noise = torch.randn_like(param) * noise_level * self.epsilon
                param.data.add_(noise)

    def train(self, game_env, iterations=200, noise_level=0.05, max_time=10):

        best_reward = 10
        best_model = self
        rewards_history = []
        epsilon_history = []
        cars = len(game_env.cars)
        noisy_models = []
        model_sigs = []

        for i in tqdm(range(iterations), desc="Training Progress", ncols=100):
            for n in range(cars):
                noisy_model = copy.deepcopy(best_model)
                noisy_model.add_noise_to_model(noise_level)
                noisy_models.append(noisy_model)

            rewards = self.evaluate_model(game_env, noisy_models, max_time=max_time)
            best_noisy_reward = max(rewards.values())
            best_noisy_model_index = max(rewards, key=rewards.get)

            if best_noisy_reward > best_reward:
                best_reward = best_noisy_reward
                best_model = noisy_models[best_noisy_model_index]
                print(f"New best model found at iteration {i} with reward: {best_reward}")


            rewards_history.append(best_noisy_reward)
            epsilon_history.append(noisy_models[best_noisy_model_index].epsilon)

            self.decay_epsilon()

            noisy_models.clear()
            model_sigs.clear()

        self.plot_var(rewards_history, 'reward history')
        self.plot_var(epsilon_history, 'epsilon history')

        return best_model

    def evaluate_model(self, game_env, noisy_models,  max_time):
        start_time = time.time()
        game_env.reset()
        no_action =2
        done = False

        cars = game_env.cars
        no_actions = {i:  no_action for i in range(len(cars))}
        rewards = {i: 0 for i in range(len(cars))}
        images_flat, scalars_flat = game_env.step(no_actions)

        while not done:
            if time.time() - start_time > max_time:
                print("\n")
                break

            actions = []
            for i, car in enumerate(cars):
                action = noisy_models[i].get_action(images_flat[i], scalars_flat[i])
                actions.append(action)
                rewards[i] = car.distance
            images_flat, scalars_flat = game_env.step(actions)

        return rewards



    def plot_var(self, var, var_name="Variable"):
        """Plot given variable with linear approximation if applicable."""
        plt.figure(figsize=(10, 6))

        # Plot the variable
        plt.plot(var, label=f"{var_name} per iteration", color='blue')

        # If it's numeric, try to fit a line
        if isinstance(var, (list, np.ndarray)) and all(isinstance(v, (int, float)) for v in var):
            x = np.arange(len(var))
            y = np.array(var)
            a, b = np.polyfit(x, y, 1)
            y_fit = a * x + b
            plt.plot(x, y_fit, label=f"Linear Fit: y = {a:.2f}x + {b:.2f}", color='red', linestyle='--')

        plt.xlabel("Iterations")
        plt.ylabel(var_name)
        plt.title(f"Training Progress: {var_name} per Iteration")
        plt.legend()
        plt.grid(True)
        plt.show()

