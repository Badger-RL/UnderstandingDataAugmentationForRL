from typing import Dict, List, Any
import numpy as np

from augment.rl.augmentation_functions.augmentation_function import AugmentationFunction

class InvertedPendulumTranslate(AugmentationFunction):

    def __init__(self,  noise_range=0.9, **kwargs):
        super().__init__()
        self.noise_range = noise_range

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
        delta = np.random.uniform(low=-self.noise_range, high=+self.noise_range, size=(n,))
        delta_x = next_obs[:,0] - obs[:,0]
        obs[:,0] = delta
        next_obs[:,0] = np.clip(delta_x + delta, -1, 1)

        return obs, next_obs, action, reward, done, infos

class InvertedPendulumReflect(AugmentationFunction):
    def __init__(self, **kwargs):
        super().__init__()

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

        obs[:,0:] *= -1
        next_obs[:,0:] *= -1
        # next_obs[:,0] -= 2*delta_x
        action *= -1

        return obs, next_obs, action, reward, done, infos

class InvertedPendulumTranslateReflect(AugmentationFunction):

    def __init__(self,  noise='uniform', **kwargs):
        super().__init__()

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
        delta = np.random.uniform(low=-0.9, high=+0.9, size=(n,))
        delta_x = next_obs[:,0] - obs[:,0]
        obs[:,0] = delta
        next_obs[:,0] = np.clip(delta_x + delta, -1, 1)

        # reflect with probability
        if np.random.random() < 0.5:
            obs[:,0:] *= -1
            next_obs[:,0:] *= -1
            action *= -1

        return obs, next_obs, action, reward, done, infos

CARTPOLE_AUG_FUNCTIONS = {
    'translate': InvertedPendulumTranslate,
    'reflect': InvertedPendulumReflect,
    'translate_reflect': InvertedPendulumTranslateReflect,
}