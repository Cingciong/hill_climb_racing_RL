import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np


class Agent(nn.Module):

    def __init__(self):
        super(Agent, self).__init__()

        self.n_actions = 3

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
        print(f"Input Scalar Shape: {input_scalar.shape}")
        print(f"Input Image Shape: {input_image.shape}")  # Should be [1, 1, H, W]

        output = self.forward(input_image, input_scalar)
        action = torch.argmax(output).item()
        print(f"Predicted Action: {action}")

        return action



    def decay_epsilon(self):
        """Decay epsilon to reduce exploration over time."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
