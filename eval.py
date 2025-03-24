import gymnasium as gym
import ale_py
from gymnasium.wrappers import AtariPreprocessing, RecordVideo
from src.agent import Agent
from src.model import NeuralNetwork
from src.train import evaluate
import torch
import os

# hyperparameters
game_index = "ALE/DemonAttack-v5"
render = "human"
recording = False

if recording: 
    # Create a directory to store the videos
    video_dir = "videos"
    os.makedirs(video_dir, exist_ok=True)
    render = "rgb_array"

# Initialize environment
env = gym.make(game_index, frameskip=1, render_mode=render)

if recording:
    # Wrap the environment with the RecordVideo wrapper
    env = gym.wrappers.RecordVideo(env, "./vid", episode_trigger=lambda episode_id: episode_id % 10 == 0)

env = AtariPreprocessing(
    env,
    noop_max=10, frame_skip=4, terminal_on_life_loss=True,
    screen_size=84, grayscale_obs=False, grayscale_newaxis=False
)   

# Initialize DQN model (agent)
num_actions = env.action_space.n
DQN = NeuralNetwork(num_actions, lr=0.01)

# Initialize PacmanAgent
agent = Agent(
    env=env,
    DQN=DQN,
    initial_epsilon=0.01,
    epsilon_decay=0.01,
    final_epsilon=0.01,
    replay_capacity=1000000,
)

# Caricamento dei pesi del modello
agent.Q.load_state_dict(torch.load("model/agent_Q.pth"))

# Evaluation
evaluate(env, agent, n_games=10)

env.close()