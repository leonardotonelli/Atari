import gymnasium as gym
import ale_py
from gymnasium.wrappers import AtariPreprocessing
from agent import Agent
from model import NeuralNetwork
import torch
from train import evaluate

# hyperparameters
game_index = "ALE/DemonAttack-v5"

# initialize environment
env = gym.make(game_index, frameskip=1, render_mode="human")
env = AtariPreprocessing(
    env,
    noop_max=10, frame_skip=4, terminal_on_life_loss=True,
    screen_size=84, grayscale_obs=False, grayscale_newaxis=False
)   

# Initialize DQN model (agent)
num_actions = env.action_space.n
DQN = NeuralNetwork(num_actions)

# Initialize PacmanAgent
agent = Agent(
    env=env,
    DQN=DQN,
    learning_rate=0.01,
    initial_epsilon=0.01,
    epsilon_decay=0.01,
    final_epsilon=0.01,
    replay_capacity=1000000,
)

# Caricamento dei pesi del modello
agent.Q.load_state_dict(torch.load("model/agent_Q.pth"))

# Evaluation
evaluate(env, agent, n_games=1)