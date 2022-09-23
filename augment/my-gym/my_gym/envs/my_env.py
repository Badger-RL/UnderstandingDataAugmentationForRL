import copy
import os

import gym
import numpy as np
import torch

from augment.rl.algs.td3 import TD3
from my_gym import ENVS_DIR


class MyEnv(gym.Env):
    def __init__(self, rbf_n, neural=False):

        self.rbf_n = rbf_n
        self.neural_features = None
        if self.rbf_n:
            self.original_obs_dim = self.observation_space.shape[-1]
            self.observation_space = gym.spaces.Box(low=-1, high=+1, shape=(2*self.rbf_n,))

            load_dir = f'{ENVS_DIR}/rbf_basis/obs_dim_{self.original_obs_dim}/n_{rbf_n}/'
            self.P = np.load(f'{load_dir}/P.npy')
            self.phi = np.load(f'{load_dir}/phi.npy')
            self.nu = np.load(f'{load_dir}/nu.npy')

            # self.P = np.random.normal(loc=0, scale=1, size=(rbf_n, self.original_obs_dim))
            # self.phi = np.random.uniform(low=-np.pi, high=np.pi, size=(rbf_n,))
            print(self.nu)

            self.obs = None
        if neural:
            print(os.getcwd())
            model = TD3.load(path='../../condor/predator_prey_time_limit/results/PredatorPrey-v0/td3/no_aug/run_0/best_model.zip', env=self)
            self.neural_features = model.actor.mu[:-2]
            self.neural_features.requires_grad_(False)
            self.neural_features.eval()
            self.original_obs_dim = self.observation_space.shape[-1]
            self.observation_space = gym.spaces.Box(low=0, high=+1, shape=(self.neural_features[0].weight.shape[0],))


    def _get_obs(self):
        if self.rbf_n:
            return self._rbf(self.obs)
        elif self.neural_features:
            obs = torch.from_numpy(self.obs).float()
            features = self.neural_features(obs)
            # print(features.detach().numpy())
            return features.detach().numpy()
        else:
            return self.obs

    def _rbf(self, obs):
        sin = np.sin(self.P.dot(obs)/np.sqrt(self.rbf_n) + self.phi)
        cos = np.cos(self.P.dot(obs)/np.sqrt(self.rbf_n) + self.phi)
        return np.concatenate((sin, cos))

    def _rbf_inverse(self, rbf_obs):
        return np.atanh(self.P.dot(rbf_obs)/self.nu + self.phi)
