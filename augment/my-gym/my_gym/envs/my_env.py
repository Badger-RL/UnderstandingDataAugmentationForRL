import copy
import os

import gym
import numpy as np

from my_gym import ENVS_DIR


class MyEnv(gym.Env):
    def __init__(self, rbf_n):

        self.rbf_n = rbf_n
        if self.rbf_n:
            original_obs_dim = self.observation_space.shape[-1]
            self.observation_space = gym.spaces.Box(low=-1, high=+1, shape=(self.rbf_n,))

            print(os.getcwd())
            print(ENVS_DIR)
            load_dir = f'{ENVS_DIR}/rbf_basis/obs_dim_{original_obs_dim}/n_{rbf_n}/'
            self.P = np.load(f'{load_dir}/P.npy')
            self.phi = np.load(f'{load_dir}/phi.npy')
            self.nu = np.load(f'{load_dir}/nu.npy')
            print(self.nu)

            self.obs = None


    def _get_obs(self):
        if self.rbf_n:
            return self._rbf(self.obs)
        else:
            return self.obs

    def _rbf(self, obs):
        return np.tanh(self.P.dot(obs)/1)

    def _rbf_inverse(self, rbf_obs):
        return np.atanh(self.P.dot(rbf_obs)/self.nu + self.phi)
