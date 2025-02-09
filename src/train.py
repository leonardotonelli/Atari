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


