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

n_episodes = 3000
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
    replay_capacity=replay_capacity,
    min_replay_size = 50000
)

# train the agent
training(env, agent, n_episodes=n_episodes, batch_size=batch_size, C=50000, verbose=(False, 0))

# Salvataggio dei pesi del modello
torch.save(agent.Q.state_dict(), "model/agent_Q.pth")
