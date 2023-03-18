from typing import Dict, List, Any
import numpy as np

from augment.rl.augmentation_functions.augmentation_function import AugmentationFunction

class CartPoleTranslate(AugmentationFunction):

    def __init__(self,  noise_scale=4, **kwargs):
        super().__init__()
        self.noise_scale = noise_scale
        print(locals())

    def _augment(self,
                obs: np.ndarray,
                next_obs: np.ndarray,
                action: np.ndarray,
                reward: np.ndarray,
                done: np.ndarray,
                infos: List[Dict[str, Any]],
                p=None,
                ):

        n = obs.shape[0]
        delta = np.random.uniform(low=-self.noise_scale, high=+self.noise_scale, size=(n,))
        delta_x = next_obs[:,0] - obs[:,0]
        obs[:,0] = delta
        next_obs[:,0] = delta_x + delta

        return obs, next_obs, action, reward, done, infos

class CartPoleReflect(AugmentationFunction):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _augment(
            self,
            obs: np.ndarray,
            next_obs: np.ndarray,
            action: np.ndarray,
            reward: np.ndarray,
            done: np.ndarray,
            infos: List[Dict[str, Any]],
            **kwargs,
    ):
        delta_x = next_obs[:,0] - obs[:,0]

        obs[:,1:] *= -1
        next_obs[:,1:] *= -1
        next_obs[:, 0] -= 2*delta_x

        mask = action == 0
        action[mask] = 1
        action[~mask] = 0
        # action = ~action

        return obs, next_obs, action, reward, done, infos

class CartPoleTranslateReflect(AugmentationFunction):

    def __init__(self,  noise_scale=0.9, **kwargs):
        super().__init__(**kwargs)
        self.noise_scale = noise_scale
        print(locals())

    def _augment(self,
                obs: np.ndarray,
                next_obs: np.ndarray,
                action: np.ndarray,
                reward: np.ndarray,
                done: np.ndarray,
                infos: List[Dict[str, Any]],
                p=None,
                ):

        n = obs.shape[0]
        delta = np.random.uniform(low=-self.noise_scale, high=+self.noise_scale, size=(n,))
        delta_x = next_obs[:,0] - obs[:,0]
        obs[:,0] = delta
        next_obs[:,0] = delta_x + delta

        obs[:,0:] *= -1
        next_obs[:,0:] *= -1
        mask = action == 0
        action[mask] = 1
        action[~mask] = 0
        return obs, next_obs, action, reward, done, infos

CARTPOLE_AUG_FUNCTIONS = {
    'translate': CartPoleTranslate,
    'reflect': CartPoleReflect,
    'translate_reflect': CartPoleTranslateReflect,
}