import time
from typing import Dict, List, Any

import numpy as np

from augment.rl.augmentation_functions.augmentation_function import AugmentationFunction

'''

    | Num | Observation                                      | Min  | Max | Name (in corresponding XML file) | Joint | Unit                     |
    | --- | ------------------------------------------------ | ---- | --- | -------------------------------- | ----- | ------------------------ |
    | 0   | z-coordinate of the top (height of hopper)       | -Inf | Inf | rootz (torso)                    | slide | position (m)             |
    | 1   | angle of the top                                 | -Inf | Inf | rooty (torso)                    | hinge | angle (rad)              |
    | 2   | angle of the thigh joint                         | -Inf | Inf | thigh_joint                      | hinge | angle (rad)              |
    | 3   | angle of the leg joint                           | -Inf | Inf | leg_joint                        | hinge | angle (rad)              |
    | 4   | angle of the foot joint                          | -Inf | Inf | foot_joint                       | hinge | angle (rad)              |
    | 5   | angle of the left thigh joint                    | -Inf | Inf | thigh_left_joint                 | hinge | angle (rad)              |
    | 6   | angle of the left leg joint                      | -Inf | Inf | leg_left_joint                   | hinge | angle (rad)              |
    | 7   | angle of the left foot joint                     | -Inf | Inf | foot_left_joint                  | hinge | angle (rad)              |
    | 8   | velocity of the x-coordinate of the top          | -Inf | Inf | rootx                            | slide | velocity (m/s)           |
    | 9   | velocity of the z-coordinate (height) of the top | -Inf | Inf | rootz                            | slide | velocity (m/s)           |
    | 10  | angular velocity of the angle of the top         | -Inf | Inf | rooty                            | hinge | angular velocity (rad/s) |
    | 11  | angular velocity of the thigh hinge              | -Inf | Inf | thigh_joint                      | hinge | angular velocity (rad/s) |
    | 12  | angular velocity of the leg hinge                | -Inf | Inf | leg_joint                        | hinge | angular velocity (rad/s) |
    | 13  | angular velocity of the foot hinge               | -Inf | Inf | foot_joint                       | hinge | angular velocity (rad/s) |
    | 14  | angular velocity of the thigh hinge              | -Inf | Inf | thigh_left_joint                 | hinge | angular velocity (rad/s) |
    | 15  | angular velocity of the leg hinge                | -Inf | Inf | leg_left_joint                   | hinge | angular velocity (rad/s) |
    | 16  | angular velocity of the foot hinge               | -Inf | Inf | foot_left_joint                  | hinge | angular velocity (rad/s) |
    ### Rewards
'''
class Walker2dReflect(AugmentationFunction):

    def __init__(self, sigma=0.1, k=4, **kwargs):
        super().__init__()
        self.sigma = sigma
        self.k = k


    def augment(self,
                augmentation_n: int,
                obs: np.ndarray,
                next_obs: np.ndarray,
                action: np.ndarray,
                reward: np.ndarray,
                done: np.ndarray,
                infos: List[Dict[str, Any]],
                delta = None,
                p=None
                ):

        aug_obs, aug_next_obs, aug_action, aug_reward, aug_done, aug_infos = self._deepcopy_transition(
            augmentation_n, obs, next_obs, action, reward, done, infos)

        rpos = obs[:, 2:5]
        lpos = obs[:, 5:8]
        rvel = obs[:, 11:14]
        lvel = obs[:, 14:]

        aug_obs[:,2:5] = lpos
        aug_obs[:,5:8] = rpos
        aug_obs[:,11:14] = lvel
        aug_obs[:,14:] = rvel

        ra = action[:, :3]
        la = action[:, 3:]

        aug_action[:, :3] = la
        aug_action[:, 3:] = ra

        return aug_obs, aug_next_obs, aug_action, aug_reward, aug_done, aug_infos


import gym, my_gym

if __name__ == "__main__":

    k=3
    env = gym.make('Walker2d-v3')
    obs = env.reset()

    x = 0 #np.pi/4
    qpos = np.copy(env.sim.data.qpos)
    # +1 since qpos includes x position
    qpos[3] = 0.5
    qpos[6] = -0.5

    qvel = np.zeros(9)
    env.set_state(qpos, qvel)
    obs = env.get_obs()
    obs = obs.reshape(1, -1)

    f = Walker2dReflect()

    action = np.zeros(6).reshape(1, -1)

    for t in range(1000):
        aug_obs, aug_next_obs, aug_action, aug_reward, aug_done, aug_infos = f.augment(1, obs, obs, action, obs, obs, [{}])

        env.set_state(qpos, qvel)

        env.render()
        time.sleep(0.001)

        qpos_aug = np.copy(qpos)
        qpos_aug[1:] = aug_obs[0, :8]

        env.set_state(qpos_aug, qvel)

        env.render()
        time.sleep(0.001)


    env.set_state(qpos, qvel)

    action = np.array([1,1,1,-1,-1,-1])
    next_obs, r, done, info = env.step(action)
    aug_obs, aug_next_obs, aug_action, aug_reward, aug_done, aug_infos = f.augment(1, obs, next_obs, action, obs, obs, [{}])

    env.set_state(qpos, qvel)
    next_obs, r, done, info = env.step(np.concatenate(action[3:], action[3:]))
