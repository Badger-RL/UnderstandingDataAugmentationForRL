
import gym
import numpy as np


class ChainEnv(gym.Env):

    def __init__(self, k=10):
        super().__init__()

        self.step_counter = 0
        self.horizon = k
        self.k = k
        self.observation_space = gym.spaces.Discrete(k+1)
        self.action_space = gym.spaces.Discrete(2)

    def step(self, a):

        self.step_counter += 1
        # print(a)
        if a == 1:
            if self.state == self.k:
                self.state = self.k
            else:
                self.state += 1
        else:
            if self.state == 0:
                self.state = 0
            else:
                self.state -= 1

        reward = 1 if self.state == self.k else 0
        done = self.step_counter == self.horizon
        return self.state, reward, done, {}

    def reset(self):
        self.step_counter = 0
        self.state = 0
        return 0

if __name__ == "__main__":
    pass