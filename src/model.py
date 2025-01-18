import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralNetwork(nn.Module):
    def __init__(self, num_actions):
        super(NeuralNetwork, self).__init__()

        # First convolutional layer: 32 filters of 8x8, stride 4
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=8, stride=4)
        
        # Second convolutional layer: 64 filters of 4x4, stride 2
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        
        # Third convolutional layer: 64 filters of 3x3, stride 1
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        # Fully connected hidden layer with 512 rectifier units
        self.fc1 = nn.Linear(in_features=64 * 7 * 7, out_features=512)

        # Output layer: one output per action
        self.out = nn.Linear(in_features=512, out_features=num_actions)

    def forward(self, x):
        # Pass input through the first convolutional layer
        x = F.relu(self.conv1(x))

        # Pass through the second convolutional layer
        x = F.relu(self.conv2(x))

        # Pass through the third convolutional layer
        x = F.relu(self.conv3(x))

        # Flatten the output from the convolutional layers
        x = x.view(x.size(0), -1)

        # Pass through the fully connected hidden layer
        x = F.relu(self.fc1(x))

        # Pass through the output layer
        x = self.out(x)
        return x

    def step(self, y, action):
        """
        Perform a gradient step given the target y and the action from which
        we are evaluating our current estimate.
        
        Args:
            y (torch.Tensor): Target values.
            action (torch.Tensor): Actions for which Q-values were computed.
        """
        # Compute the loss
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)

        # Forward pass
        current_q_values = self.forward(action)

        # Calculate the loss
        loss = loss_fn(current_q_values, y)

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()
    
    def get_value(self, state, action):
        q_values = self.forward(state)
        value = q_values[action]
        return value

# Example of usage:
# Assuming input image dimensions are (3, 84, 84) and 4 valid actions
num_actions = 4
model = NeuralNetwork(num_actions)

# Example input tensor of batch size 1
example_input = torch.rand(1, 3, 84, 84)  # Batch size 1, RGB channels

# Forward pass
output = model(example_input)
print(output)
