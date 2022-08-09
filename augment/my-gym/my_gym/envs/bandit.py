import os

import gym
import numpy as np
from matplotlib import pyplot as plt


class BanditEnv(gym.Env):

    def __init__(self, n=1, mu=None, sparse=False, exponent=0.5):
        super().__init__()

        self.step_counter = 0

        self.observation_space = gym.spaces.Discrete(1)
        self.action_space = gym.spaces.Box(low=-1, high=+1, shape=(n,))
        self.n = n
        self.sparse = sparse
        self.exponent = exponent

        if mu is None:
            self.mu = np.ones(n)*0.5
        else:
            assert self.n == len(mu)
            self.mu = mu

    def step(self, a):

        # a = self._action_map(a)
        # a = np.sin(a*np.pi/2)
        dist = np.linalg.norm(a-self.mu)
        if self.sparse:
            reward = 1 if dist < 0.01 else 0
        else:
            reward = -dist**2
        done = True

        # self.step_counter += 1

        return 0, reward, done, {}

    def reset(self):
        return 0

    def _gaussian(self, mu, sigma, x):
        return np.exp(-np.power(x - mu, 2) / (2 * sigma**2))

    def _action_map(self, a):

        # return self._gaussian(mu=self.mu, sigma=0.1, x=a)
        if self.exponent % 2 == 1:
            a = a**self.exponent
        else:
            neg = a < 0
            pos = a > 0
            a[pos] = a[pos]**self.exponent
            a[neg] = -(-a[neg])**self.exponent
        return a

if __name__ == "__main__":

    def gaussian(x, mu, sig):
        return 1/np.sqrt(2*np.pi*sig**2) * np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

    x = np.linspace(-1, 1, 100)
    y = 1/2 + x**2
    area = 1 + 2/3
    plt.plot(x, y/area)
    plt.ylim(0, 2)
    plt.show()

    a = np.linspace(-1, 1, 1000)
    exponent = 5
    mu = 0

    z = np.empty_like(a)
    neg = a < 0
    pos = a > 0
    z[pos] = a[pos] ** exponent
    z[neg] = -(-a[neg]) ** exponent

    def f(x):
        return (x-mu)**2
    plt.scatter(a, f(a))
    plt.scatter(z, f(z))
    plt.scatter(a, f(z))
    plt.show()

    plt.hist(a, bins=100, alpha=0.2, density=True)
    values, bins, _ = plt.hist(z, bins=100, alpha=0.2, density=True)
    print(bins.shape, values.shape)
    area = sum(np.diff(bins[70:81]) * values[70:80])
    print(area)
    plt.show()