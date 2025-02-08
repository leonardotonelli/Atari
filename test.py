import gymnasium as gym
import numpy as np
import random
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim

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
        self.reward_memory = 250
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
        rewards = torch.clamp(rewards, -1.0, 1.0) # clio rewards between 1 and -1
        
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





from matplotlib import pyplot as plt
from tqdm import tqdm
from PIL import Image
import numpy as np
import gymnasium as gym
import time
import os
import heapq
from PIL import Image
import cv2


def training(env, agent, n_episodes: int, batch_size: int, C: int, verbose = (False, 0)):
    """
    Train the agent in the environment for a specified number of episodes.

    Args:
        env: The environment to train the agent in.
        agent: The agent to be trained.
        n_episodes: Number of episodes to train for.
        batch_size: Size of the minibatch for training.
    """
    
    rep = 0 # counter to track the delayed update of Q
    episode_rewards = []  # To track rewards per episode
    episode_Q_values = [] # To track average Q-value per episode

    for episode in tqdm(range(n_episodes)):
        obs, info = env.reset()  # Reset the environment
        done = False
        total_reward = 0  # Track total reward for the episode
        stateavg_Q_values = [] # Track state-averaged Q-values for the episode
        
        if verbose[0]:
            print("New Game has started!")

        while not done:
            rep += 1
            if rep == C:
                # Update the target Q-network
                agent.update_Q_at()

            # select next action
            action = agent.get_action(obs)

            if verbose[0]:
                print(f"This is the action just chosen: {action}")
                time.sleep(verbose[1])

            # Take action in the environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if verbose[0]:
                print(f"This is the reward from the action just played: {reward}")
                print("")
            
            # Store experience in memory
            agent.store_memory(obs, action, reward, next_obs, terminated)

            # Update the agent's Q-network using replay memory
            agent.update_Q(batch_size)
            
            # to avoid getting stuck in a 0 sequence loop
            if len(agent.rewards) > 400 and sum(agent.rewards) == 0:
                terminated = True

            # Update `done` and the current state
            done = terminated or truncated
            obs = next_obs
            
            # Save average Q-value for the episode
            action_Q_value = agent.Q_values.detach().cpu().numpy()
            stateavg_Q_values.append(action_Q_value.mean()) # Log the episode's average Q-value
        
        # Decay exploration rate
        agent.decay_epsilon()

        # Log the episode's reward
        episode_rewards.append(total_reward)
        
        # Log the episode's average Q-value
        avg_q = np.mean(stateavg_Q_values) if stateavg_Q_values else 0
        episode_Q_values.append(float(avg_q))

        if verbose[0]:
            print("-"*10)
            print(f"Episode just ended, total reward: {total_reward}, average q-value: {avg_q}")
            print("-"*10)

    # Plot the training progress
    plt.plot(episode_rewards)
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title("Training Progress")
    plt.show()
    
    # Plot the Q-values
    plt.plot(episode_Q_values)
    plt.xlabel("Episodes")
    plt.ylabel("Average Q-values")
    plt.title("Q-values Progress")
    plt.show()
    
    
def save_obs(obs, path= "frames/highest.png"):
    # Convert to a PIL Image and save
    image = Image.fromarray(obs)
    image.save(path)  # Save as PNG


def evaluate(env, agent, n_games = 10):
    agent.epsilon = 0.01
    episode_rewards = []
    highest_value, highest_frame = -np.inf, 0
    lowest_value, lowest_frame = np.inf, 0
    for episode in tqdm(range(n_games)):
        obs, info = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            # store interesting frames
            action = agent.get_action(obs)
            if agent.Q_values[action] > highest_value:
                highest_frame, highest_value = obs, agent.Q_values[action]
            if agent.Q_values[action] < lowest_value:
                lowest_frame, lowest_value = obs, agent.Q_values[action]
            
            # Take action in the environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            # Update `done` and the current state
            done = terminated or truncated
            obs = next_obs


        episode_rewards.append(total_reward)

    # Plot the training progress
    plt.plot(episode_rewards)
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title("Training Progress")
    plt.show()

    save_obs(highest_frame)
    save_obs(lowest_frame, "frames/lowest.png")


