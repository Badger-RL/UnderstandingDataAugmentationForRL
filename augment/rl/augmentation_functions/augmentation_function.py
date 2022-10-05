from copy import deepcopy
from typing import Dict, List, Any

import numpy as np

from my_gym import ENVS_DIR


class AugmentationFunction:

    def __init__(self, env=None, **kwargs):
        self.env = env

    def _deepcopy_transition(
            self,
            augmentation_n: int,
            obs: np.ndarray,
            next_obs: np.ndarray,
            action: np.ndarray,
            reward: np.ndarray,
            done: np.ndarray,
            infos: List[Dict[str, Any]],
    ):
        aug_obs = deepcopy(obs).repeat(augmentation_n, axis=0)
        aug_next_obs = deepcopy(next_obs).repeat(augmentation_n, axis=0)
        aug_action = deepcopy(action).repeat(augmentation_n, axis=0)
        aug_reward = deepcopy(reward).repeat(augmentation_n, axis=0)
        aug_done = deepcopy(done).repeat(augmentation_n, axis=0)
        aug_infos = [deepcopy([infos[i]]*augmentation_n) for i in range(len(infos))]

        return aug_obs, aug_next_obs, aug_action, aug_reward, aug_done, aug_infos

    def augment(self,
                 aug_n: int,
                 obs: np.ndarray,
                 next_obs: np.ndarray,
                 action: np.ndarray,
                 reward: np.ndarray,
                 done: np.ndarray,
                 infos: List[Dict[str, Any]],
                 **kwargs,):

        v = self._augment(aug_n, *self._deepcopy_transition(aug_n, obs, next_obs, action, reward, done, infos))
        return v
    def _augment(self,
                 aug_n: int,
                 obs: np.ndarray,
                 next_obs: np.ndarray,
                 action: np.ndarray,
                 reward: np.ndarray,
                 done: np.ndarray,
                 infos: List[Dict[str, Any]],
                 **kwargs,):
        raise NotImplementedError("Augmentation function not implemented.")

if __name__ == "__main__":

    f = AugmentationFunction(rbf_n=16)

    m = 1000
    observations = np.random.uniform(-1, +1, size=(m, 4))
    rbf_observations = []
    for i in range(m):
        rbf_observations.append(f.rbf(observations[i]))
    rbf_observations = np.array(rbf_observations)

    inv_rbf_observations = f.rbf_inverse(rbf_observations)

    assert np.allclose(observations, inv_rbf_observations)

