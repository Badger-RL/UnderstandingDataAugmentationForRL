import gym
import numpy as np

from augment.rl.augmentation_functions import PredatorPreyRotate, PredatorPreyTranslateDense, AUGMENTATION_FUNCTIONS
from augment.simulate import simulate

def make_assertions(aug_next_obs, aug_reward, aug_done, aug_info,
                    next_obs_true, reward_true, done_true, info_true):

    assert np.allclose(aug_next_obs, next_obs_true)
    assert np.allclose(aug_reward, reward_true)

    # aug_info != info_true in general.
    dist = np.linalg.norm(aug_next_obs[:2] - aug_next_obs[2:])
    assert aug_done == ((dist < 0.05) or (aug_info == {'TimeLimit.truncated': True}))

def test_augmentation(env, augmentation_function):
    for ep in range(100):
        observations, next_observations, actions, rewards, dones, infos = simulate(model=None, env=env, num_episodes=1, seed=0, render=False, flatten=True, verbose=0)
        aug_observations, aug_next_observations, aug_actions, aug_rewards, aug_dones, aug_infos = augmentation_function.augment(3, observations, next_observations, actions, rewards, dones, infos)

        for aug_obs, aug_next_obs, aug_action, aug_reward, aug_done, aug_info in zip(aug_observations, aug_next_observations, aug_actions, aug_rewards, aug_dones, aug_infos):
            env.reset()
            env.set_state(np.copy(aug_obs[:2]), np.copy(aug_obs[2:]))
            next_obs_true, reward_true, done_true, info_true = env.step(aug_action)

            make_assertions(aug_next_obs, aug_reward, aug_done, aug_info,
                    next_obs_true, reward_true, done_true, info_true)

if __name__ == "__main__":

    for env_id, aug_function_classes in AUGMENTATION_FUNCTIONS.items():
        for aug_function_name, aug_function_class in aug_function_classes.items():
            print(env_id, aug_function_name)
            if 'Predator' in env_id:
                env = gym.make(env_id)
                aug_function = aug_function_class()
                test_augmentation(env, aug_function)


