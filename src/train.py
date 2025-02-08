from matplotlib import pyplot as plt
from tqdm import tqdm
from PIL import Image
import numpy as np
import gymnasium as gym
import time
import os
import heapq
from PIL import Image
import cv2

# Save frames
highest_q_frames = []  # Max heap for highest Q-values
lowest_q_frames = []   # Min heap for lowest Q-values

def save_frame(frame, q_value, frame_type, episode, step):
    os.makedirs("saved_frames", exist_ok=True)
    file_name = f"saved_frames/{frame_type}_q{q_value:.2f}_ep{episode}_step{step}.png"
    cv2.imwrite(file_name, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    
    # Convert frame (NumPy array) to an image and save
    image = Image.fromarray(frame)
    image.save(file_name)


def update_q_frame_records(frame, q_value, episode, step):
    
    frame = tuple(map(tuple, frame))

    # Maintain max-heap for highest Q-values (store negative Q-values for heapq to work correctly)
    if len(highest_q_frames) < 4:
        heapq.heappush(highest_q_frames, (-q_value, frame, episode, step))
    else:
        heapq.heappushpop(highest_q_frames, (-q_value, frame, episode, step))

    # Maintain min-heap for lowest Q-values
    if len(lowest_q_frames) < 4:
        heapq.heappush(lowest_q_frames, (q_value, frame, episode, step))
    else:
        heapq.heappushpop(lowest_q_frames, (q_value, frame, episode, step))


def save_final_q_frames():
    # Save highest Q-value frames
    for neg_q_value, frame, episode, step in highest_q_frames:
        save_frame(frame, -neg_q_value, "highest", episode, step)

    # Save lowest Q-value frames
    for q_value, frame, episode, step in lowest_q_frames:
        save_frame(frame, q_value, "lowest", episode, step)


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
            if rep == C:
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
            
            if len(agent.rewards) > 200  and sum(agent.rewards) == 0:
                terminated = True
            # Update `done` and the current state
            done = terminated or truncated
            obs = next_obs
            
            # Save average Q-value for the episode
            action_Q_value = agent.Q_values.detach().cpu().numpy()
            stateavg_Q_values.append(action_Q_value.mean()) # Log the episode's average Q-value
        
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

    # Plot the training progress
    plt.plot(episode_rewards)
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title("Training Progress")
    plt.show()
    
    # Plot the Q-values
    plt.plot(episode_Q_values)
    plt.xlabel("Episodes")
    plt.ylabel("Average Q-values")
    plt.title("Q-values Progress")
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
            if agent.Q_values[action] > highest_value:
                highest_frame, highest_value = obs, agent.Q_values[action]
            if agent.Q_values[action] < lowest_value:
                lowest_frame, lowest_value = obs, agent.Q_values[action]
            
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


