import gymnasium as gym
import numpy as np
import random
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import copy

class Agent:
    def __init__(
        self,
        env: gym.Env,
        DQN: nn.Module,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.99,
        replay_capacity: int = 500000,
        min_replay_size: int = 10000
    ):
        self.env = env
        self.Q = DQN  
        self.Q_at = copy.deepcopy(DQN)  

        self.discount_factor = discount_factor
        self.memory_capacity = replay_capacity
        self.min_replay_size = min_replay_size
        self.memory = []
        self.reward_memory = 5000
        self.rewards = []

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

        # reward clipping
        reward = np.clip(reward, -1.0, 1.0) 

        if len(self.memory) >= self.memory_capacity:
            self.memory.pop(0)
        self.memory.append((current_state, action, reward, next_state, terminated))

        if len(self.rewards) >= self.reward_memory:
            self.rewards.pop(0)
        self.rewards.append(reward)

    def sample_memory(self, batch_size):
        """Return a random minibatch from the memory."""
        return random.sample(self.memory, batch_size)

    def update_Q(self, batch_size: int):
        """
        Updates the parameters of the DQN using a random batch from memory.
        """
        
        # First check if we have enough samples
        if len(self.memory) < self.min_replay_size:
            return False
        
        #sample minibatch from the memory
        if len(self.memory) < batch_size:
            return  # Not enough samples to update
    

        batch = self.sample_memory(batch_size)
        
        # create the batch dataset
        current_states = torch.tensor(np.array([sample[0] for sample in batch]), dtype=torch.float32)
        current_states = current_states.permute(0, 3, 1, 2)  # Reorder dimensions
        actions = torch.tensor(np.array([sample[1] for sample in batch]), dtype=torch.int64)  
        rewards = torch.tensor(np.array([sample[2] for sample in batch]), dtype=torch.float32)
        
        next_states = torch.tensor(np.array([sample[3] for sample in batch]), dtype=torch.float32)
        next_states = next_states.permute(0, 3, 1, 2)  # From [batch, height, width, channels] to [batch, channels, height, width]

        terminated = torch.tensor(np.array([sample[4] for sample in batch]), dtype=torch.float32)

        temp = self.Q_at(next_states).max(dim=1).values

        current_q_values = self.Q(current_states)  # Shape: [batch_size, num_actions]
        actions = actions.unsqueeze(1)  # Shape: [batch_size, 1]
        outputs = current_q_values.gather(1, actions).squeeze(1)  # Shape: [batch_size]

        targets = rewards + self.discount_factor * (1 - terminated) * temp

        # make the gradient step and record training error
        loss = self.Q.step(targets, outputs)  
        self.training_error.append(loss)

    def update_Q_at(self):
        """Update the target network to match the main network."""
        self.Q_at.load_state_dict(self.Q.state_dict())

    def decay_epsilon(self):
        """Decay the exploration rate."""
        self.epsilon = max(self.final_epsilon, self.epsilon * self.epsilon_decay)
