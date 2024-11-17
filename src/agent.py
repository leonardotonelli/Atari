import gymnasium as gym
import numpy as np

class PacmanAgent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
        replay_capacity: int = 1000,
        DQN = "dqn" #TODO
        ):

        self.env = env
        self.Q = DQN #initialize the DQN
        self.Q_at = DQN
        self.Q_values = np.zeros(env.action_space.n)
        self.Q_at_values = np.zeros(env.action_space.n)

        self.lr = learning_rate
        self.discount_factor = discount_factor
        self.memory_capacity = replay_capacity
        self.memory = []

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []


    def compute_q_values(self, state: np.array):
        self.Q_values = self.Q.eval(state)

    def get_action(self) -> int:
        """
        Returns the best action with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.
        """
        # with probability epsilon return a random action to explore the environment
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        # with probability (1 - epsilon) act greedily (exploit)
        else:
            return int(np.argmax(self.Q_values))
    
    def store_memory(self, current_state, action, reward, next_state):
        if len(self.memory)+1 == self.memory_capacity:
            self.memory = []
        else:
            self.memory.append((current_state, action, reward, next_state))

    def sample_memory(self):
        """ return a random minibatch from the memory: tuple (current_state, action, reward, next_state)"""
        return self.memory[np.random.choice(range(len(self.memory)))]

    def update_Q(
            self,
            current_state: np.array,
            action: int,
            reward: float,
            terminated: bool,
            next_state: np.array,
    ):
        """Updates the parameters of the DQN, given a random-batch (current_state, action, reward, next_state)"""
        y = reward + (not terminated) * self.discount_factor * np.max(self.Q_at.eval(next_state)) 
        error = ( y - self.Q_values[action] )^2 
        self.training_error.append(error)
        self.Q.step(y, action) #TODO

    def update_Q_at(self):
        """Updates the parameters of the DQN, given a random-batch (current_state, action, reward, next_state)"""
        self.Q_at = self.Q

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)


## methods for DQN:
##  .forward() : forward pass for a given input state
##  .step(y, action) : make a gradient step given the target y and the action from which we are evaluating our current estimate