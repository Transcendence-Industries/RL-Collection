import random
import numpy as np
import gymnasium as gym
from datetime import datetime
from tqdm import tqdm
from gymnasium.spaces import MultiDiscrete

from utils.logging import MLFlow_Logger
from utils.file_handling import save_pickle

"""
Environment: https://gymnasium.farama.org/environments/box2d/lunar_lander/

Observations:
[0] - x position (-1.5; 1.5)
[1] - y position (-1.5; 1.5)
[2] - x velocity (-5; 5)
[3] - y velocity (-5; 5)
[4] - angle (-3.1415927; 3.1415927)
[5] - angular velocity (-5; 5)
[6] - left leg ground contact (0; 1)
[7] - right leg ground contact (0; 1)

Actions:
0 - do nothing
1 - fire left engine
2 - fire center engine
3 - fire right engine
"""


class LunarLander_Agent:
    def __init__(self, render=False):
        # Environment
        self.env = gym.make(
            "LunarLander-v2",
            render_mode="human" if render else None,
            continuous=False,
            gravity=-10.0,
            enable_wind=False,
            wind_power=15.0,
            turbulence_power=1.5)

        # Observation/Action space
        discrete_observation_interval = np.array((20, 20, 10, 10, 5, 10, 1, 1), dtype=np.int8)
        self.discrete_observation_window_size = ((self.env.observation_space.high - self.env.observation_space.low)
                                                 / discrete_observation_interval)
        discrete_observation_space = MultiDiscrete(discrete_observation_interval)
        action_space = self.env.action_space

        # Q-table
        self.q_table = np.zeros((*(discrete_observation_interval + 1), action_space.n), dtype=np.float32)
        print(f"Observation space: {discrete_observation_space}")
        print(f"Action space: {action_space}")
        print(f"Shape/Size of Q-table: {self.q_table.shape} / {self.q_table.size}")

        # Logging
        self.logger = MLFlow_Logger(experiment=self.__class__.__name__)

        print("Agent is ready!")
        print(30 * "_")

    def get_discrete_observation(self, obs):
        obs = np.clip(obs, self.env.observation_space.low, self.env.observation_space.high)
        discrete_obs = (obs - self.env.observation_space.low) / self.discrete_observation_window_size
        return discrete_obs.astype(np.int8)

    def value_in_tolerance(self, value, target, tolerance):
        return target - tolerance <= value <= target + tolerance

    def train(self, n_episodes, learning_rate, gamma, min_epsilon, max_epsilon, epsilon_decay):
        # Logging
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.logger.create_run(run=timestamp)
        self.logger.log_parameters({"n_episodes": n_episodes,
                                    "learning_rate": learning_rate,
                                    "gamma": gamma,
                                    "min_epsilon": min_epsilon,
                                    "max_epsilon": max_epsilon,
                                    "epsilon_decay": epsilon_decay})
        self.logger.log_metric(key="epsilon", value=max_epsilon, step=0)
        self.logger.log_metric(key="reward", value=0, step=0)

        for episode in tqdm(range(1, n_episodes + 1)):
            # print(f"Episode: {episode}")

            # Reset environment
            state, _ = self.env.reset()
            state = self.get_discrete_observation(state)

            # Set parameters
            epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-epsilon_decay * episode)
            self.logger.log_metric(key="epsilon", value=epsilon, step=episode)
            episode_reward = 0
            end_episode = False

            while not end_episode:
                # Calculate best or pick random action
                if random.uniform(0, 1) > epsilon:
                    action = np.argmax(self.q_table[state])
                else:
                    action = self.env.action_space.sample()

                # Execute action
                new_state, reward, terminated, truncated, _ = self.env.step(action)
                new_state = self.get_discrete_observation(new_state)
                episode_reward += reward

                # Calculate Q-values
                self.q_table[state][action] = self.q_table[state][action] + learning_rate * (
                        float(reward) + gamma * np.max(self.q_table[new_state]) - self.q_table[state][action])
                state = new_state

                # Check for goal
                if (self.value_in_tolerance(state[0], 0, 0.1)
                        and self.value_in_tolerance(state[1], 0, 0.1)):
                    print("Goal reached!")
                    end_episode = True

                # Check for episode end
                if terminated or truncated:
                    end_episode = True

            self.logger.log_metric(key="reward", value=episode_reward, step=episode)

        # Close environment and save model
        print(30 * "_")
        print("Agent shutdown.")
        self.env.close()
        self.logger.end_run()
        save_pickle(model=self.q_table, experiment=self.__class__.__name__, run=timestamp)

    def run(self, q_table, n_steps):
        if q_table.shape != self.q_table.shape:
            raise Exception(f"The given Q-table has not the desired shape of {self.q_table.shape}!")

        self.q_table = q_table
        state, _ = self.env.reset()
        state = self.get_discrete_observation(state)

        for step in range(n_steps):
            action = np.argmax(self.q_table[state])
            new_state, reward, terminated, truncated, _ = self.env.step(action)
            state = self.get_discrete_observation(new_state)

        self.env.close()
