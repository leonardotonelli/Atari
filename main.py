import numpy as np
import gymnasium as gym
import ale_py
from gymnasium.wrappers import AtariPreprocessing
from tqdm import tqdm
from src.agent import PacmanAgent
from src.training import training
from model import NeuralNetwork

gym.register_envs(ale_py)

# hyperparameters
game_index = "ALE/MsPacman-v5"
learning_rate = 0.01
initial_epsilon = 1.0
n_episodes = 100
epsilon_decay = initial_epsilon / (n_episodes / 2)  # reduce the exploration over time
final_epsilon = 0.3
replay_capacity = 10000

env = gym.make(game_index, frameskip=1)
env = AtariPreprocessing(
    env,
    noop_max=10, frame_skip=4, terminal_on_life_loss=True,
    screen_size=84, grayscale_obs=False, grayscale_newaxis=False
)   
num_actions = env.action_space.n
state_shape = env.observation_space.shape

# Initialize DQN model
DQN = NeuralNetwork(num_actions)

# Initialize PacmanAgent
agent = PacmanAgent(
    env=env,
    DQN=DQN,
    learning_rate=learning_rate,
    initial_epsilon=initial_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
    replay_capacity=replay_capacity,
)

# Train the agent
training(env, agent, n_episodes=10, batch_size=64)



