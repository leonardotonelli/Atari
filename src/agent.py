import gymnasium as gym
import numpy as np
import random
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

class Agent:
    def __init__(
        self,
        env: gym.Env,
        DQN: nn.Module,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
        replay_capacity: int = 1000
    ):
        self.env = env
        self.Q = DQN  
        self.Q_at = DQN  

        self.optimizer = optim.Adam(self.Q.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

        self.discount_factor = discount_factor
        self.memory_capacity = replay_capacity
        self.memory = []

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []


    def get_action(self, state) -> int:
        """
        Returns the best action with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.
        """
        input_tensor = torch.tensor(state, dtype=torch.float32)
        state = input_tensor.unsqueeze(0).permute(0, 3, 1, 2)
        self.Q_values = self.Q(state)
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return int(torch.argmax(self.Q_values))

    def store_memory(self, current_state, action, reward, next_state, terminated):
        """Store experiences in replay memory."""
        if len(self.memory) >= self.memory_capacity:
            self.memory.pop(0)
        self.memory.append((current_state, action, reward, next_state, terminated))

    def sample_memory(self, batch_size):
        """Return a random minibatch from the memory."""
        return random.sample(self.memory, batch_size)

    def update_Q(self, batch_size: int):
        """
        Updates the parameters of the DQN using a random batch from memory.
        """
        #sample minibatch from the memory
        if len(self.memory) < batch_size:
            return  # Not enough samples to update
        batch = self.sample_memory(batch_size)
        # print(f"This is sample types: {type(batch[0][0])}")
        # create the batch dataset
        current_states = torch.tensor(np.array([sample[0] for sample in batch]), dtype=torch.float32)
        current_states = current_states.permute(0, 3, 1, 2)  # Reorder dimensions
        actions = torch.tensor(np.array([sample[1] for sample in batch]), dtype=torch.int64)  
        rewards = torch.tensor(np.array([sample[2] for sample in batch]), dtype=torch.float32)
        next_states = torch.tensor(np.array([sample[3] for sample in batch]), dtype=torch.float32)
        next_states = next_states.permute(0, 3, 1, 2)  # From [batch, height, width, channels] to [batch, channels, height, width]

        terminated = torch.tensor(np.array([sample[4] for sample in batch]), dtype=torch.float32)

        temp = torch.stack([
            self.Q_at.forward(next_state).detach() for next_state in next_states
        ]).max(dim=1).values

        targets = rewards + terminated * temp
        outputs = torch.tensor(np.array([self.Q.get_value(current_state, current_action) for current_state, current_action in zip(current_states, actions)]), requires_grad=True)
        
        # make the gradient step and record training error
        loss = self.Q.step(targets.to(torch.float64), outputs.to(torch.float64))  

        # store the loss in the class field
        self.training_error.append(loss)

    def update_Q_at(self):
        """Update the target network to match the main network."""
        self.Q_at.load_state_dict(self.Q.state_dict())

    def decay_epsilon(self):
        """Decay the exploration rate."""
        self.epsilon = max(self.final_epsilon, self.epsilon * self.epsilon_decay)
