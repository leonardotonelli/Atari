import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np

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

        self.optimizer = torch.optim.RMSprop(self.parameters(), lr=0.001, alpha=0.99, eps=1e-08, weight_decay=0)
        self.loss_fn = torch.nn.HuberLoss(reduction='mean', delta=1.0) # error/gradient clipping

    def forward(self, x):
        # print("Input shape:", x.shape)
        x = F.relu(self.conv1(x))
        # print("After conv1:", x.shape)
        x = F.relu(self.conv2(x))
        # print("After conv2:", x.shape)
        x = F.relu(self.conv3(x))
        # print("After conv3:", x.shape)
        x =  torch.flatten(x)  # Flatten
        # print("After flatten:", x.shape)
        x = F.relu(self.fc1(x))
        # print("After fc1:", x.shape)
        x = self.out(x)
        return x

    def step(self, targets, outputs):
        """
        Perform a gradient step given a mini_batch. This function will update the
        DQN object permorming a gradient step in the direction of the targets. 
        Inputs are two tensors defining targets and the outputs of the DQN.
        """
        loss = self.loss_fn(outputs, targets)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss.item()
    
    def get_value(self, state, action=None):
        q_values = self(state)
        value = q_values[action]
        return float(value)
