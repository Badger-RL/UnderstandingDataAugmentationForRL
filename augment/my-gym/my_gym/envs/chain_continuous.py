
import gym
from my_gym.envs import ChainEnv


class ChainContinuousEnv(ChainEnv):

    def __init__(self, k=10):
        super().__init__(k=k)
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(1,))

    def step(self, a):
        a = self._convert_to_to_discrete(a)
        return super().step(a)

    def _convert_to_to_discrete(self, a):
        if a < 0.5:
            return 0
        else:
            return 1

if __name__ == "__main__":
    pass