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

    def __init__(self, **kwargs):
        super().__init__()

    def _augment(self,
                obs: np.ndarray,
                next_obs: np.ndarray,
                action: np.ndarray,
                reward: np.ndarray,
                done: np.ndarray,
                infos: List[Dict[str, Any]],
                p=None
                ):

        rpos = obs[:, 3:5].copy()
        lpos = obs[:, 6:8].copy()
        rvel = obs[:, 12:14].copy()
        lvel = obs[:, 15:].copy()
        obs[:,3:5] = lpos
        obs[:,6:8] = rpos
        obs[:,12:14] = lvel
        obs[:,15:] = rvel

        rpos = obs[:, 3:5].copy()
        lpos = obs[:, 6:8].copy()
        rvel = obs[:, 12:14].copy()
        lvel = obs[:, 15:].copy()
        next_obs[:,3:5] = lpos
        next_obs[:,6:8] = rpos
        next_obs[:,12:14] = lvel
        next_obs[:,15:] = rvel

        ra = action[:, :3].copy()
        la = action[:, 3:].copy()

        action[:, :3] = la
        action[:, 3:] = ra

        return obs, next_obs, action, reward, done, infos


if __name__ == "__main__":
    env = gym.make('Walker2d-v3')

    f = Walker2dReflect()

    env.reset()
    # set initial qpos, qvel
    qpos = np.copy(env.data.qpos)
    qvel = np.copy(env.data.qvel)


    # +1 index since qpos includes x position
    # qpos[3] = np.random.uniform(-1, +1)
    # qpos[6] = np.random.uniform(-1, +1)
    # qpos[2] = -np.pi/4
    qpos[3] = -np.pi/4
    qpos[4] = -np.pi/4

    # qpos[5] = np.pi/4
    qpos[6] = np.pi/4
    # qpos[2] = -np.pi/4
    qpos[7] = np.pi/4

    obs = np.concatenate([qpos[1:], qvel])
    # qpos = np.ones(9)
    # qvel = np.ones(9)

    for i in range(100):
        env.reset()
        # # set initial qpos, qvel
        # qpos = np.copy(env.sim.data.qpos)
        # qvel = np.copy(env.sim.data.qvel)
        #
        # # +1 index since qpos includes x position
        # qpos[3] = np.random.uniform(-1, +1)
        # qpos[6] = np.random.uniform(-1, +1)
        env.set_state(qpos, qvel)
        obs = env.get_obs()

        # get transition
        action = np.random.uniform(-1,1,6)
        action = np.ones(6)*1
        next_obs, reward, done, info = env.step(action)
        obs = obs.reshape(1, -1)
        next_obs = next_obs.reshape(1, -1)
        action = action.reshape(1, -1)
        done = np.array([done]).reshape(1, -1)

        env.reset()
        # env.set_state(qpos, qvel)

        env.render()
        time.sleep(0.1)

        aug_obs, aug_next_obs, aug_action, aug_reward, aug_done, aug_infos = f.augment(1, obs, next_obs, action, reward, done, [{}])

        # Make sure aug transition matches simulation
        # aug_obs to qpos, qvel
        qpos2, qvel2 = env.obs_to_q(aug_obs[0])
        env.set_state(qpos2, qvel2)
        obs2 = env.get_obs()
        next_obs2, reward2, done2, info2 = env.step(aug_action[0])

        env.render()
        time.sleep(0.1)
        # print(aug_next_obs)
        # print(next_obs2)


        print(aug_obs - obs2)
        print(aug_next_obs - next_obs2)
        print(aug_reward - reward2, aug_reward, reward2)

        # assert np.allclose(aug_obs, obs2)
        # assert np.allclose(aug_next_obs, next_obs2)
        # assert np.allclose(aug_reward, reward2)
#
# if __name__ == "__main__":
#
#     k=3
#     env = gym.make('Walker2d-v3')
#     obs = env.reset()
#
#     x = 0 #np.pi/4
#     qpos = np.copy(env.sim.data.qpos)
#     # +1 since qpos includes x position
#     qpos[3] = 0.5
#     qpos[6] = -0.5
#
#     qvel = np.zeros(9)
#     env.set_state(qpos, qvel)
#     obs = env.get_obs()
#     obs = obs.reshape(1, -1)
#
#     f = Walker2dReflect()
#
#     action = np.zeros(6).reshape(1, -1)
#
#     for t in range(1000):
#         aug_obs, aug_next_obs, aug_action, aug_reward, aug_done, aug_infos = f.augment(1, obs, obs, action, obs, obs, [{}])
#
#         env.set_state(qpos, qvel)
#
#         env.render()
#         time.sleep(0.001)
#
#         qpos_aug = np.copy(qpos)
#         qpos_aug[1:] = aug_obs[0, :8]
#
#         env.set_state(qpos_aug, qvel)
#
#         env.render()
#         time.sleep(0.001)
#
#
#     env.set_state(qpos, qvel)
#
#     action = np.array([1,1,1,-1,-1,-1])
#     next_obs, r, done, info = env.step(action)
#     aug_obs, aug_next_obs, aug_action, aug_reward, aug_done, aug_infos = f.augment(1, obs, next_obs, action, obs, obs, [{}])
#
#     env.set_state(qpos, qvel)
#     next_obs, r, done, info = env.step(np.concatenate(action[3:], action[3:]))
