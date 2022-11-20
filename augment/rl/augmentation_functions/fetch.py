from typing import Dict, List, Any

import numpy as np
import torch
from matplotlib import pyplot as plt

from augment.rl.augmentation_functions.augmentation_function import AugmentationFunction
# import gym, my_gym

'''
    - `observation`: its value is an `ndarray` of shape `(10,)`. It consists of kinematic information of the end effector. The elements of the array correspond to the following:
        | Num | Observation                                                                                                                           | Min    | Max    | Site Name (in corresponding XML file) | Joint Name (in corresponding XML file) |Joint Type| Unit                     |
        |-----|---------------------------------------------------------------------------------------------------------------------------------------|--------|--------|---------------------------------------|----------------------------------------|----------|--------------------------|
        | 0   | End effector x position in global coordinates                                                                                         | -Inf   | Inf    | robot0:grip                           |-                                       |-         | position (m)             |
        | 1   | End effector y position in global coordinates                                                                                         | -Inf   | Inf    | robot0:grip                           |-                                       |-         | position (m)             |
        | 2   | End effector z position in global coordinates                                                                                         | -Inf   | Inf    | robot0:grip                           |-                                       |-         | position (m)             |
        | 3   | Joint displacement of the right gripper finger                                                                                        | -Inf   | Inf    |-                                      | robot0:r_gripper_finger_joint          | hinge    | position (m)             |
        | 4   | Joint displacement of the left gripper finger                                                                                         | -Inf   | Inf    |-                                      | robot0:l_gripper_finger_joint          | hinge    | position (m)             |
        | 5   | End effector linear velocity x direction                                                                                              | -Inf   | Inf    | robot0:grip                           |-                                       |-         | velocity (m/s)           |
        | 6   | End effector linear velocity y direction                                                                                              | -Inf   | Inf    | robot0:grip                           |-                                       |-         | velocity (m/s)           |
        | 7   | End effector linear velocity z direction                                                                                              | -Inf   | Inf    | robot0:grip                           |-                                       |-         | velocity (m/s)           |
        | 8   | Right gripper finger linear velocity                                                                                                  | -Inf   | Inf    |-                                      | robot0:r_gripper_finger_joint          | hinge    | velocity (m/s)           |
        | 9   | Left gripper finger linear velocity                                                                                                   | -Inf   | Inf    |-                                      | robot0:l_gripper_finger_joint          | hinge    | velocity (m/s)           |
        
'''

class FetchReachAugmentationFunction(AugmentationFunction):

    def __init__(self, env,  **kwargs):
        super().__init__(env=env, **kwargs)
        self.env = env
        self.lo = env.initial_gripper_xpos[:3] - env.target_range
        self.hi = env.initial_gripper_xpos[:3] + env.target_range
        self.delta = 0.05
        self.desired_mask = env.desired_mask
        self.achieved_mask = env.achieved_mask

class FetchReachHER(FetchReachAugmentationFunction):

    def __init__(self, env,  **kwargs):
        super().__init__(env=env, **kwargs)

    def _augment(self,
                 obs: np.ndarray,
                 next_obs: np.ndarray,
                 action: np.ndarray,
                 reward: np.ndarray,
                 done: np.ndarray,
                 infos: List[Dict[str, Any]],
                 p=None,
                 ):

        obs[:, self.desired_mask] = obs[-1, self.achieved_mask]
        done = self.env.is_success(obs[:, self.achieved_mask], obs[:, self.desired_mask])
        reward = self.env.compute_reward(obs[:, self.achieved_mask], obs[:, self.desired_mask], infos)

        end = np.argmax(done)+1
        obs = obs[:end]
        next_obs = next_obs[:end]
        action = action[:end]
        reward = reward[:end]
        done = done[:end]
        infos = infos[:end]


        return obs, next_obs, action, reward, done, infos

class FetchReachTranslate(FetchReachAugmentationFunction):

    def __init__(self, env, **kwargs):
        super().__init__(env=env, **kwargs)
        self.env = env
        self.lo = env.initial_gripper_xpos[:3] - env.target_range
        self.hi = env.initial_gripper_xpos[:3] + env.target_range
        self.delta = 0.05
        self.distance_threshold = self.env.distance_threshold

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
        action *= -1
        obs[:,5:8] *= -1
        next_obs[:,5:8] *= -1

        next_obs[:, :3] = -2*(next_obs[:, self.achieved_mask] - obs[:, :3])
        dist = np.linalg.norm(next_obs[:, :3] - next_obs[:, -3:], axis=-1)
        reward[:] = dist < self.distance_threshold


        return obs, next_obs, action, reward, done, infos