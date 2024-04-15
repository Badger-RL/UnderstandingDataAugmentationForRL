import time
from typing import Dict, List, Any
import numpy as np
import gym

from augment.rl.augmentation_functions.augmentation_function import AugmentationFunction


class SwimmerReflect(AugmentationFunction):

    def __init__(self, sigma=0.1, k=2, **kwargs):
        super().__init__()
        self.sigma = sigma
        self.k = k


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
        k = 3
        obs[:,:k] *= -1
        obs[:,-k:] *= -1
        obs[:,k+1] *= -1

        next_obs[:,:k] *= -1
        next_obs[:,-k:] *= -1
        next_obs[:,k+1] *= -1


        action *= -1

        return obs, next_obs, action, reward, done, infos


def check_valid(env, aug_obs, aug_next_obs, aug_action, aug_reward, aug_done, aug_info):

    # set env to aug_obs
    # env = gym.make('Walker2d-v4', render_mode='human')

    # env.reset()
    qpos, qvel = env.obs_to_q(aug_obs)
    x_pos = np.array([0])
    qpos = np.concatenate([x_pos, qpos])
    env.set_state(qpos, qvel)

    # determine ture next_obs, reward
    next_obs_true, reward_true, terminated_true, truncated_true, info_true = env.step(aug_action)
    print(aug_next_obs - next_obs_true)
    assert np.allclose(aug_next_obs, next_obs_true)
    assert np.allclose(aug_reward, reward_true)

def sanity_check():
    env = gym.make('Swimmer-v4', render_mode='human')
    env.reset()

    f = SwimmerReflect(k=2)

    qpos_orginal = env.data.qpos.copy()
    qvel_original = env.data.qvel.copy()

    qpos = qpos_orginal.copy()
    qvel = qvel_original.copy()
    print(qpos)
    print(qvel)
    env.set_state(qpos, qvel)

    action = np.zeros(2)
    action[0] = +1
    action[1] = -1
    for j in range(20):
        next_obs, _, _, _, _ = env.step(action)
    print(next_obs)
    print()
    env.render()
    time.sleep(1)

SWIMMER_AUG_FUNCTIONS = {
    'reflect': SwimmerReflect,
}



if __name__ == "__main__":
    sanity_check()

    # env = gym.make('Swimmer-v4', render_mode='human')
    # aug_func = SwimmerReflect(k=2)
    #
    # validate_augmentation(env, aug_func, check_valid)
