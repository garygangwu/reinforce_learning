import numpy as np
from collections import defaultdict
import pickle



class QLearningAgent:
    def __init__(
        self,
        env,
        learning_rate: float = 0.01,
        initial_epsilon: float = 1,
        epsilon_decay: float = 0.0001,
        final_epsilon: float = 0.01,
        discount_factor: float = 0.95,
    ):
        """Initialize a Reinforcement Learning agent with an empty dictionary
        of state-action values (q_values), a learning rate and an epsilon.

        Args:
            learning_rate: The learning rate
            initial_epsilon: The initial epsilon value
            epsilon_decay: The decay for epsilon
            final_epsilon: The final epsilon value
            discount_factor: The discount factor for computing the Q-value
        """
        self.env = env
        self.q_values = defaultdict(self.default_value)

        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []

    def default_value(self):
        return np.zeros(self.env.action_space.n)
    
    def get_action_from_q_result(self, obs) -> int:
        return int(np.argmax(self.q_values[obs]))
        
    def get_action(self, obs) -> int:
        """
        Returns the best action with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.
        """
        # with probability epsilon return a random action to explore the environment
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()

        # with probability (1 - epsilon) act greedily (exploit)
        else:
            return self.get_action_from_q_result(obs)

    def update(
        self,
        obs,
        action: int,
        reward: float,
        terminated: bool,
        next_obs,
    ):
        """Updates the Q-value of an action."""
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        #future_q_value = np.max(self.q_values[next_obs])
        temporal_difference = (
            reward + self.discount_factor * future_q_value - self.q_values[obs][action]
        )

        self.q_values[obs][action] = (
            self.q_values[obs][action] + self.lr * temporal_difference
        )
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

    def save_to_file(self, filename: str):
        with open(filename, 'wb') as file:
            pickle.dump(self.q_values, file)

    def load_from_file(self, filename: str):
        with open(filename, 'rb') as file:
            self.q_values = pickle.load(file)
