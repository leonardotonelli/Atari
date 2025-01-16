import gymnasium as gym
import numpy as np

import numpy as np
import random
import gym
import torch
import torch.nn as nn
import torch.optim as optim

class PacmanAgent:
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
        self.Q = DQN  # Initialize the main DQN
        self.Q_at = DQN  # Initialize the target DQN

        self.optimizer = optim.Adam(self.Q.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

        self.discount_factor = discount_factor
        self.memory_capacity = replay_capacity
        self.memory = []

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []

    def compute_q_values(self, state: np.array):
        """Compute Q-values for a given state."""
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        self.Q_values = self.Q(state_tensor).detach().numpy()[0]

    def get_action(self) -> int:
        """
        Returns the best action with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.
        """
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return int(np.argmax(self.Q_values))

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
        if len(self.memory) < batch_size:
            return  # Not enough samples to update

        batch = self.sample_memory(batch_size)
        
        current_states = torch.tensor([sample[0] for sample in batch], dtype=torch.float32)
        actions = torch.tensor([sample[1] for sample in batch], dtype=torch.int64)
        rewards = torch.tensor([sample[2] for sample in batch], dtype=torch.float32)
        next_states = torch.tensor([sample[3] for sample in batch], dtype=torch.float32)
        terminated = torch.tensor([sample[4] for sample in batch], dtype=torch.bool)

        # Compute Q-values for current states
        q_values = self.Q(current_states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute target Q-values
        with torch.no_grad():
            next_q_values = self.Q_at(next_states).max(1)[0]
            targets = rewards + (1 - terminated.float()) * self.discount_factor * next_q_values

        # Compute loss
        loss = self.loss_fn(q_values, targets)
        self.training_error.append(loss.item())

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_Q_at(self):
        """Update the target network to match the main network."""
        self.Q_at.load_state_dict(self.Q.state_dict())

    def decay_epsilon(self):
        """Decay the exploration rate."""
        self.epsilon = max(self.final_epsilon, self.epsilon * self.epsilon_decay)



## methods for DQN:
##  .forward() : forward pass for a given input state
##  .step(y, action) : make a gradient step given the target y and the action from which we are evaluating our current estimate