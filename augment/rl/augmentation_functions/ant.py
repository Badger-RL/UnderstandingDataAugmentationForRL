import time
from typing import Dict, List, Any
import numpy as np
import gym

# from augment.rl.augmentation_functions import validate_augmentation
from augment.rl.augmentation_functions.augmentation_function import AugmentationFunction


class AntReflect(AugmentationFunction):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.obs_permute = np.arange(27)
        # joint angles
        self.obs_permute[5] = 7
        self.obs_permute[6] = 8
        self.obs_permute[7] = 5
        self.obs_permute[8] = 6
        self.obs_permute[9] = 11
        self.obs_permute[10] = 12
        self.obs_permute[11] = 9
        self.obs_permute[12] = 10
        # joint vels
        self.obs_permute[19] = 21
        self.obs_permute[20] = 22
        self.obs_permute[21] = 19
        self.obs_permute[22] = 20
        self.obs_permute[23] = 25
        self.obs_permute[24] = 26
        self.obs_permute[25] = 23
        self.obs_permute[26] = 24

        self.obs_reflect = np.zeros(27, dtype=bool)
        self.obs_reflect[5:12+1] = True
        self.obs_reflect[13] = True
        self.obs_reflect[17] = True
        self.obs_reflect[18] = True
        self.obs_reflect[19:] = True


        # self.obs_mask_left = np.zeros(27, dtype=bool)
        # # pos
        # self.obs_mask_left[5:6+1] = True
        # self.obs_mask_left[9:10+1] = True
        # # vel
        # self.obs_mask_left[19:20+1] = True
        # self.obs_mask_left[23:24+1] = True
        #
        # self.obs_mask_right = np.zeros(27, dtype=bool)
        # # pos
        # self.obs_mask_right[7:8+1] = True
        # self.obs_mask_right[11:12+1] = True
        # # vel
        # self.obs_mask_right[21:22+1] = True
        # self.obs_mask_right[25:26+1] = True

        # torso, reflect y values
        # self.obs_mask_reflect_y = np.zeros(27, dtype=bool)
        # self.obs_mask_reflect_y[14] = True
        # self.obs_mask_reflect_y[17] = True

        # thigh left:  0, 2
        # thigh right: 6, 4
        self.action_permute = np.arange(8)
        self.action_permute[0] = 6
        self.action_permute[2] = 4
        self.action_permute[4] = 2
        self.action_permute[6] = 0

        self.action_permute[1] = 7 #-
        self.action_permute[3] = 5 #-
        self.action_permute[5] = 3 #-
        self.action_permute[7] = 1 #-

        # self.action_reflect = np.zeros(8, dtype=bool)
        # self.action_mask_right[2:3+1] = True
        # self.action_mask_right[4:5+1] = True



    def _swap_obs_left_right(self, obs):
        left = obs[:, 5:].copy()
        right = obs[:, self.obs_mask_right].copy()
        obs[:, self.obs_mask_right] = -left
        obs[:, self.obs_mask_left] = -right


        # left = obs[:, self.obs_mask_left].copy()
        # right = obs[:, self.obs_mask_right].copy()
        # obs[:, self.obs_mask_right] = -left
        # obs[:, self.obs_mask_left] = -right
        # obs[:,6] *= -1
        # obs[:,8] *= -1
        # obs[:,10] *= -1
        # obs[:,12] *= -1

    def _swap_action_left_right(self, action):
        # left = action[:, self.action_mask_left].copy()
        # right = action[:, self.action_mask_right].copy()
        # action[:, self.action_mask_right] = -left
        # action[:, self.action_mask_left] = -right

        action[:, :] = action[:, self.action_permute]
        action[:, :] *= -1

    def _reflect_y(self, obs):
        obs[:, self.obs_mask_reflect_y] *= -1

    def _reflect_orientation(self, obs):
        obs[:, 3] *= -1
        obs[:, 4] *= -1

        # qy = obs[:, 2].copy()
        # qz = obs[:, 3].copy()
        # obs[:, 2] = -qz
        # obs[:, 3] = qy

    def _augment(self,
                obs: np.ndarray,
                next_obs: np.ndarray,
                action: np.ndarray,
                reward: np.ndarray,
                done: np.ndarray,
                infos: List[Dict[str, Any]],
                delta = None,
                p=None
                ):

        # k = (obs.shape[-1]-2)//2
        # self._swap_obs_left_right(obs)
        # self._swap_obs_left_right(next_obs)
        #
        # self._reflect_y(obs)
        # self._reflect_y(next_obs)
        #
        # self._reflect_orientation(obs)
        # self._reflect_orientation(next_obs)

        obs[:, :] = obs[:, self.obs_permute]
        next_obs[:, :] = next_obs[:, self.obs_permute]

        obs[:, self.obs_reflect] *= -1
        next_obs[:, self.obs_reflect] *= -1
        self._reflect_orientation(obs)
        self._reflect_orientation(next_obs)

        self._swap_action_left_right(action)
        reward_forward = infos[0][0]['reward_forward']
        reward[:] += -2*reward_forward

        return obs, next_obs, action, reward, done, infos



class AntRotate(AugmentationFunction):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _augment(self,
                obs: np.ndarray,
                next_obs: np.ndarray,
                action: np.ndarray,
                reward: np.ndarray,
                done: np.ndarray,
                infos: List[Dict[str, Any]],
                delta = None,
                p=None
                ):


        obs[:, 3] += 0.1
        next_obs[:, 4] += 0.1

        reward_forward = infos[0]['reward_forward']
        reward[:] += -2*reward_forward

        return obs, next_obs, action, reward, done, infos


def check_valid(env, aug_obs, aug_next_obs, aug_action, aug_reward, aug_done, aug_info):

    # set env to aug_obs
    # env = gym.make('Walker2d-v4', render_mode='human')

    # env.reset()
    qpos, qvel = aug_obs[:12+1], aug_obs[12+1:]
    x = aug_info['x_position']
    y = aug_info['y_position']
    qpos = np.concatenate((np.array([0,0]), qpos))
    env.set_state(qpos, qvel)

    # determine ture next_obs, reward
    next_obs_true, reward_true, terminated_true, truncated_true, info_true = env.step(aug_action)
    # print(aug_next_obs - next_obs_true)
    print('here', aug_reward-reward_true)
    assert np.allclose(aug_next_obs, next_obs_true)
    # assert np.allclose(aug_reward, reward_true)

def sanity_check():
    env = gym.make('Ant-v4', reset_noise_scale=0)
    env.reset()

    f = AntRotate()

    qpos = env.data.qpos.copy()
    qvel = env.data.qvel.copy()
    env.set_state(qpos, qvel)

    action = np.zeros(8)
    action[0] = 0.5
    action[2] = -1

    for j in range(100):
        next_obs, _, _, _, _ = env.step(action)

    true = next_obs.copy()

    ####

    env.set_state(qpos, qvel)

    action = np.zeros(8)
    action[6] = -0.5
    action[4] = 1

    for j in range(100):
        next_obs, reward, terminated, truncated, info = env.step(action)

    print('a2')
    for i in range(27):
        print(f'{i}\t{true[i]:.8f}\t{next_obs[i]:.8f}')
    print()

    obs = next_obs.copy().reshape(1,-1)
    next_obs = next_obs.reshape(1,-1)
    action = action.reshape(1,-1)
    infos = np.array([info])
    rewards = np.array([reward])
    aug_obs, aug_next_obs, x,x,x,x = f._augment(obs, next_obs, action, rewards, 0, infos)

    print(true)
    print()

    print(aug_next_obs[0])

    print()
    print(aug_next_obs[0]-true)
    is_close = np.isclose(aug_next_obs[0],true)
    # print(aug_next_obs[:, ~is_close])


ANT_AUG_FUNCTIONS = {
    'reflect': AntReflect,
    'rotate': AntRotate,
}



if __name__ == "__main__":
    sanity_check()
    '''
    
[ 3.72480987e-01  9.87922126e-01  9.22158027e-14  5.22475962e-14
 1.32880515e-01  5.23557528e-01  5.26546683e-01
 -5.23557528e-01  1.32880513e-01 -5.23557528e-01  5.26546683e-01
  5.23557528e-01  1.55274708e-15 -2.07514643e-15 -5.06126326e-16
  3.44954146e-16 -1.52314561e-17  1.60550302e-15  4.35959755e-15
  2.95143465e-15 -2.14390850e-15 -1.88547344e-15 -8.68566744e-15
 -6.22520524e-16 -1.91939428e-15  1.74032731e-15]
 
 [ 3.72480987e-01  9.87922126e-01  1.86673410e-14 -3.29259099e-14
 5.26546683e-01  5.23557528e-01  1.32880514e-01
 -5.23557528e-01  5.26546683e-01 -5.23557528e-01  1.32880515e-01
  5.23557528e-01 -7.53279354e-16 -6.12279030e-16  5.25870224e-16
 -8.21463135e-17 -6.64805014e-18  1.60969211e-15 -2.06311139e-15
  1.64066752e-15 -4.61133029e-15 -1.24168551e-15 -2.00795122e-15
 -2.13531251e-15  2.77747207e-16  2.17558401e-15]
    '''
    env = gym.make('Ant-v4', reset_noise_scale=0)
    aug_func = AntRotate()
    # validate_augmentation(env, aug_func, check_valid)
