from LunarLander.agent import LunarLander_Agent
from utils.file_handling import load_pickle
from utils.threading import MultiCore_Controller


def simple_train():
    agent = LunarLander_Agent(render=True)
    agent.train(n_episodes=1_000,
                learning_rate=0.1,
                gamma=0.95,
                min_epsilon=0.05,
                max_epsilon=1,
                epsilon_decay=0.0005)


def multicore_train_wrapper(n_episodes, learning_rate, gamma, min_epsilon, max_epsilon, epsilon_decay):
    agent = LunarLander_Agent(render=True)
    agent.train(n_episodes=n_episodes,
                learning_rate=learning_rate,
                gamma=gamma,
                min_epsilon=min_epsilon,
                max_epsilon=max_epsilon,
                epsilon_decay=epsilon_decay)


def multicore_train():
    params = [
        (10,  # n_episodes
         0.1,  # learning_rate
         0.95,  # gamma
         0.05,  # min_epsilon
         1,  # max_epsilon
         0.0005),  # epsilon_decay
        (10,  # n_episodes
         0.5,  # learning_rate
         0.95,  # gamma
         0.05,  # min_epsilon
         1,  # max_epsilon
         0.0005)  # epsilon_decay
    ]

    controller = MultiCore_Controller(n_cores=4)
    controller.run(function=multicore_train_wrapper, param_list=params)


def simple_run():
    model = load_pickle(experiment="LunarLander_Agent", run="2024-04-08_22-47-01")
    agent = LunarLander_Agent(render=True)
    agent.run(model, n_steps=1_000)
