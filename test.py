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
        replay_capacity: int = 500000
    ):
        self.env = env
        self.Q = DQN  
        self.Q_at = copy.deepcopy(DQN)  

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
        
        next_states = torch.tensor(np.array([sample[3] for sample in batch]), dtype=torch.float32)
        next_states = next_states.permute(0, 3, 1, 2)  # From [batch, height, width, channels] to [batch, channels, height, width]

        terminated = torch.tensor(np.array([sample[4] for sample in batch]), dtype=torch.float32)

        temp = self.Q_at(next_states).max(dim=1).values

        current_q_values = self.Q(current_states)  # Shape: [batch_size, num_actions]
        actions = actions.unsqueeze(1)  # Shape: [batch_size, 1]
        outputs = current_q_values.gather(1, actions).squeeze(1)  # Shape: [batch_size]

        targets = rewards + self.discount_factor * (1 - terminated) * temp
        # outputs = torch.tensor(np.array([self.Q.get_value(current_state, current_action) for current_state, current_action in zip(current_states, actions)]), requires_grad=True)
        
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

class NeuralNetwork(nn.Module):
    def __init__(self, num_actions, lr):
        super(NeuralNetwork, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=64 * 7 * 7, out_features=512)
        self.out = nn.Linear(in_features=512, out_features=num_actions)

        self.optimizer = torch.optim.RMSprop(self.parameters(), lr=lr, alpha=0.99, eps=1e-08, weight_decay=0)
        self.loss_fn = torch.nn.HuberLoss(reduction='mean', delta=1.0) # error/gradient clipping

    def forward(self, x):
        # print("Input shape:", x.shape)
        x = F.relu(self.conv1(x))
        # print("After conv1:", x.shape)
        x = F.relu(self.conv2(x))
        # print("After conv2:", x.shape)
        x = F.relu(self.conv3(x))
        # print("After conv3:", x.shape)
        x = torch.flatten(x, start_dim=1)  # Flatten
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
import time
from PIL import Image
import seaborn as sns


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
    training_error = []

    for episode in tqdm(range(n_episodes)):
        obs, info = env.reset()  # Reset the environment
        done = False
        total_reward = 0  # Track total reward for the episode
        stateavg_Q_values = [] # Track state-averaged Q-values for the episode
        
        if verbose[0]:
            print("New Game has started!")

        while not done:
            rep += 1
            if rep % C == 0:
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

            # Save average Q-value for the episode
            action_Q_value = agent.Q_values.detach().cpu().numpy()
            stateavg_Q_values.append(action_Q_value.mean()) # Log the episode's average Q-value
        
            # Update `done` and the current state
            done = terminated or truncated
            obs = next_obs
            
        # Decay exploration rate
        agent.decay_epsilon()

        # Log the episode's reward
        episode_rewards.append(total_reward)
        
        # Log training error
        training_error.append(np.array(agent.training_error).mean())

        # Log the episode's average Q-value
        avg_q = np.mean(stateavg_Q_values) if stateavg_Q_values else 0
        episode_Q_values.append(float(avg_q))

        if verbose[0]:
            print("-"*10)
            print(f"Episode just ended, total reward: {total_reward}, average q-value: {avg_q}")
            print("-"*10)

    # Set Seaborn style for a professional look
    sns.set_style("whitegrid")
    sns.set_palette("Purples_r")

    # Create a figure with subplots
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # Plot the training progress
    axes[0].plot(episode_rewards, color="#7B1FA2", linewidth=2)
    axes[0].set_xlabel("Episodes", fontsize=12, fontweight="bold", color="#4A148C")
    axes[0].set_ylabel("Total Reward", fontsize=12, fontweight="bold", color="#4A148C")
    axes[0].set_title("Training Progress", fontsize=14, fontweight="bold", color="#6A1B9A")

    # Plot the Q-values
    axes[1].plot(episode_Q_values, color="#8E24AA", linewidth=2)
    axes[1].set_xlabel("Episodes", fontsize=12, fontweight="bold", color="#4A148C")
    axes[1].set_ylabel("Average Q-values", fontsize=12, fontweight="bold", color="#4A148C")
    axes[1].set_title("Q-values Progress", fontsize=14, fontweight="bold", color="#6A1B9A")

    # Adjust layout
    plt.tight_layout()

    # Save the figure
    plot_path = "plots/training_progress.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {plot_path}")

    # Show the plot
    plt.show()


    fig2, ax2 = plt.subplots(figsize=(10, 5))

    # Plot training error
    ax2.plot(training_error, color="#D81B60", linewidth=2)
    ax2.set_xlabel("Episodes", fontsize=12, fontweight="bold", color="#880E4F")
    ax2.set_ylabel("Training Error", fontsize=12, fontweight="bold", color="#880E4F")
    ax2.set_title("Training Error Progress", fontsize=14, fontweight="bold", color="#AD1457")

    # Adjust layout and save the figure
    plt.tight_layout()
    error_plot_path = "plots/training_error.png"
    plt.savefig(error_plot_path, dpi=300, bbox_inches="tight")
    print(f"Training error plot saved to {error_plot_path}")

    # Show the training error plot
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
            if agent.Q_values[0][action] > highest_value:
                highest_frame, highest_value = obs, agent.Q_values[0][action]
            if agent.Q_values[0][action] < lowest_value:
                lowest_frame, lowest_value = obs, agent.Q_values[0][action]
            
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



import gymnasium as gym
import ale_py
from gymnasium.wrappers import AtariPreprocessing
from agent import Agent
from train import training
from model import NeuralNetwork
import torch
import numpy as np

gym.register_envs(ale_py)

# hyperparameters
game_index = "ALE/DemonAttack-v5"

n_episodes = 1000
batch_size = 32

learning_rate = 0.00025
initial_epsilon = 1
final_epsilon = .1
epsilon_decay = epsilon_decay = np.exp(np.log(final_epsilon / initial_epsilon) / n_episodes)
discount_factor = 0.99

replay_capacity = 1000000


# initialize environment
env = gym.make(game_index, frameskip=1)
env = AtariPreprocessing(
    env,
    noop_max=30, frame_skip=4, terminal_on_life_loss=True,
    screen_size=84, grayscale_obs=False, grayscale_newaxis=False
)   

num_actions = env.action_space.n
state_shape = env.observation_space.shape

# Initialize DQN model (agent)
DQN = NeuralNetwork(num_actions, learning_rate)

# Initialize PacmanAgent
agent = Agent(
    env=env,
    DQN=DQN,
    initial_epsilon=initial_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
    discount_factor = discount_factor,
    replay_capacity=replay_capacity
)

# train the agent
training(env, agent, n_episodes=n_episodes, batch_size=batch_size, C=10000, verbose=(False, 0))

# Salvataggio dei pesi del modello
torch.save(agent.Q.state_dict(), "model/agent_Q.pth")

