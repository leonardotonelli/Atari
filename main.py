import numpy as np
import gymnasium as gym
import ale_py
from gymnasium.wrappers import AtariPreprocessing
from tqdm import tqdm
from src.agent import PacmanAgent
from src.training import training, visualize_errors
gym.register_envs(ale_py)

# hyperparameters
learning_rate = 0.01
start_epsilon = 1.0
n_episodes = 100
epsilon_decay = start_epsilon / (n_episodes / 2)  # reduce the exploration over time
final_epsilon = 0.2

env = gym.make("ALE/MsPacman-v5", frameskip=1)
env = AtariPreprocessing(
    env,
    noop_max=10, frame_skip=4, terminal_on_life_loss=True,
    screen_size=84, grayscale_obs=False, grayscale_newaxis=False
)

env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes) ## see better, think later about visualization


agent = PacmanAgent(
    env=env,
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
)

training(env, agent, n_episodes)
visualize_errors(env, agent)


