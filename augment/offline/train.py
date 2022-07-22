import gym
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from augment.offline.utils import dataset_augmented, dataset_random, load_dataset, augment
from d3rlpy.algos import DQN
from d3rlpy.datasets import get_cartpole
from d3rlpy.wrappers.sb3 import SB3Wrapper


def train(model, env, dataset,
          n_epochs=50,
          n_steps=10000,
          verbose=1):

    avgs = []
    stds = []

    # Train it using for instance a dataset created by a SB3 agent (see above)
    # model.fit(dataset, n_epochs=n_epochs, verbose=verbose, save_interval=np.inf, show_progress=verbose, save_metrics=False,)
    model.fit(dataset, n_epochs=50, verbose=verbose, save_interval=np.inf, show_progress=0, save_metrics=False)

    wrapped_model = SB3Wrapper(model)
    mean_reward, std_reward = evaluate_policy(wrapped_model, env, n_eval_episodes=100)
    print(f"mean_reward={mean_reward} +/- {std_reward}")

    avgs.append(mean_reward)
    stds.append(std_reward)

    # avg = np.average(avgs)
    # std = np.sqrt(np.square(stds).sum())
    # std = np.std(avgs)
    # print(f'Offline performance: {avg} +/- {std}')

    # return avgs, stds
    return mean_reward, std_reward

if __name__ == "__main__":
    dataset, env = get_cartpole()

    env_id = 'CartPole-v1'
    env = Monitor(gym.make(env_id))
    env.reset()

    path = '../data/CartPole-v1/trained.npz'
    data = np.load(path)
    # model = DQN(learning_rate=1e-3, batch_size=1000, )
    n_trials = 10
    results = {}
    for n in [0,2,4,8,16]:
        print(f'\nn = {n}')
        n = int(n)

        results[str(n)] = {}
        avgs = []
        stds = []
        for i in range(n_trials):
            # dataset = dataset_random(n=n)
            d = augment(data, n=n)
            if n==0: n+=1
            model = DQN(learning_rate=1e-4, batch_size=256*n)
            avg, std = train(model, env, d, n_epochs=10, n_steps=5000, verbose=1)
            avgs.append(avg)
            stds.append(stds)

        print(np.average(avgs), np.std(avgs))






