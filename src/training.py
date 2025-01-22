from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
from agent import PacmanAgent
from model import NeuralNetwork
import ale_py
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing
gym.register_envs(ale_py)

def training(env, agent, n_episodes: int, batch_size: int):
    """
    Train the agent in the environment for a specified number of episodes.

    Args:
        env: The environment to train the agent in.
        agent: The agent to be trained.
        n_episodes: Number of episodes to train for.
        batch_size: Size of the minibatch for training.
    """
    episode_rewards = []  # To track rewards per episode

    for episode in tqdm(range(n_episodes)):
        obs, info = env.reset()  # Reset the environment
        done = False
        total_reward = 0  # Track total reward for the episode

        while not done:
            # # Compute Q-values and decide action
            # agent.compute_q_values(obs)
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

        # Update the target Q-network
        agent.update_Q_at()

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

env = gym.make("ALE/MsPacman-v5", frameskip=1)
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
    learning_rate=1e-4,
    initial_epsilon=1.0,
    epsilon_decay=0.99,
    final_epsilon=0.1,
    replay_capacity=10000,
)

# Train the agent
training(env, agent, n_episodes=1, batch_size=1)


def visualize_errors(env, agent):

    # visualize the episode rewards, episode length and training error in one figure
    fig, axs = plt.subplots(1, 3, figsize=(20, 8))

    # np.convolve will compute the rolling mean for 100 episodes

    axs[0].plot(np.convolve(env.return_queue, np.ones(100)))
    axs[0].set_title("Episode Rewards")
    axs[0].set_xlabel("Episode")
    axs[0].set_ylabel("Reward")

    axs[1].plot(np.convolve(env.length_queue, np.ones(100)))
    axs[1].set_title("Episode Lengths")
    axs[1].set_xlabel("Episode")
    axs[1].set_ylabel("Length")

    axs[2].plot(np.convolve(agent.training_error, np.ones(100)))
    axs[2].set_title("Training Error")
    axs[2].set_xlabel("Episode")
    axs[2].set_ylabel("Temporal Difference")

    plt.tight_layout()
    plt.show()

