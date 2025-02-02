from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
from agent import PacmanAgent
from model import NeuralNetwork
import ale_py
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing
gym.register_envs(ale_py)

def training(env, agent, n_episodes: int, batch_size: int, C: int):
    """
    Train the agent in the environment for a specified number of episodes.

    Args:
        env: The environment to train the agent in.
        agent: The agent to be trained.
        n_episodes: Number of episodes to train for.
        batch_size: Size of the minibatch for training.
    """
    episode_rewards = []  # To track rewards per episode
    s = 0 
    for episode in tqdm(range(n_episodes)):
        s+=1
        print(s)
        obs, info = env.reset()  # Reset the environment
        done = False
        total_reward = 0  # Track total reward for the episode
        rep = 0

        while not done:
            rep += 1
            if rep == C:
                # Update the target Q-network
                agent.update_Q_at()

            # select next action
            action = agent.get_action(obs)

            # Take action in the environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            # Store experience in memory
            agent.store_memory(obs, action, reward, next_obs, terminated)

            # Update the agent's Q-network using replay memory
            agent.update_Q(batch_size)

            # Update `done` and the current state
            done = terminated or truncated
            obs = next_obs

        # Decay exploration rate
        agent.decay_epsilon()

        # Log the episode's reward
        episode_rewards.append(total_reward)

    # Plot the training progress
    plt.plot(episode_rewards)
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title("Training Progress")
    plt.show()

