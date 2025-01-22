import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

class NeuralNetwork(nn.Module):
    def __init__(self, num_actions):
        super(NeuralNetwork, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=64 * 7 * 7, out_features=512)
        self.out = nn.Linear(in_features=512, out_features=num_actions)

        self.training_error = []
        

    def forward(self, x):
        print("Input shape:", x.shape)
        x = F.relu(self.conv1(x))
        print("After conv1:", x.shape)
        x = F.relu(self.conv2(x))
        print("After conv2:", x.shape)
        x = F.relu(self.conv3(x))
        print("After conv3:", x.shape)
        x =  torch.flatten(x)  # Flatten
        print("After flatten:", x.shape)
        x = F.relu(self.fc1(x))
        print("After fc1:", x.shape)
        x = self.out(x)
        return x

    def step(self, targets, outputs):
        """
        Perform a gradient step given a mini_batch. This function will update the
        DQN object permorming a gradient step in the direction of the targets. 
        Inputs are two tensors defining targets and the outputs of the DQN.
        """
        self.optimizer = torch.optim.RMSprop(self.parameters(), lr=0.001, alpha=0.99, eps=1e-08, weight_decay=0)
        loss_fn = nn.MSELoss()
        loss = loss_fn(outputs, targets)
        self.training_error.append(loss.item())

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
    
    def get_value(self, state, action=None):
        q_values = self(state)
        value = q_values[action]
        return float(value)



# Example of usage:
# Assuming input image dimensions are (3, 84, 84) and 4 valid actions
num_actions = 4
model = NeuralNetwork(num_actions)

# Example input tensor of batch size 1
example_input = torch.rand(1, 3, 84, 84)  # Batch size 1, RGB channels

# Forward pass
output = model(example_input)
print(output)
