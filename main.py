import numpy as np
import gymnasium as gym
import ale_py
from gymnasium.wrappers import AtariPreprocessing
from tqdm import tqdm
from agent import Agent
from training import training, evaluate
from model import NeuralNetwork

gym.register_envs(ale_py)

# hyperparameters
game_index = "ALE/DemonAttack-v5"

n_episodes = 2
batch_size = 1

learning_rate = 0.01
initial_epsilon = 1
epsilon_decay = initial_epsilon / (n_episodes / 2)  
final_epsilon = 0.1

replay_capacity = 10000


# initialize environment
env = gym.make(game_index, frameskip=1, render_mode="human")
env = AtariPreprocessing(
    env,
    noop_max=10, frame_skip=4, terminal_on_life_loss=True,
    screen_size=84, grayscale_obs=False, grayscale_newaxis=False
)   

num_actions = env.action_space.n
state_shape = env.observation_space.shape

# Initialize DQN model (agent)
DQN = NeuralNetwork(num_actions)

# Initialize PacmanAgent
agent = Agent(
    env=env,
    DQN=DQN,
    learning_rate=learning_rate,
    initial_epsilon=initial_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
    replay_capacity=replay_capacity,
)

# train the agent
training(env, agent, n_episodes=n_episodes, batch_size=batch_size, C=10000, verbose=(False, 2))



# evaluation
evaluate(env, agent, n_games = 2)