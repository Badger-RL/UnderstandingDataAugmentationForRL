import gym
import numpy as np

from augment.rl.augmentation_functions import AUGMENTATION_FUNCTIONS
from augment.simulate import simulate

def make_assertions(aug_next_obs, aug_reward, aug_done, aug_info,
                    next_obs_true, reward_true, done_true, info_true):

    # print(aug_next_obs, next_obs_true, next_obs_true-aug_next_obs)
    assert np.allclose(aug_next_obs, next_obs_true, atol=1e-7)

    # aug_info != info_true in general.
    dist = np.linalg.norm(aug_next_obs[2:] - aug_next_obs[:2])
    # print(aug_reward, reward_true, dist)
    assert np.allclose(aug_reward, reward_true)

    # print(aug_done, done_true, dist, aug_info, info_true)

    # ONLY TEST IF INFO IS ACTUALLY CHANGED BY AUG FUNCTION
    if aug_done:
        if dist < 0.05:
            assert aug_info == {'TimeLimit.truncated': False}
        else:
            assert aug_info == {'TimeLimit.truncated': True}
    else:
        assert aug_info == {}

def test_augmentation(env, augmentation_function):
    aug_n = 1
    num_success = 0
    num_failure = 0
    num_success_true = 0

    for ep in range(100):
        observations, next_observations, actions, rewards, dones, infos = simulate(model=None, env=env, num_episodes=1, seed=ep, render=False, flatten=True, verbose=0)
        observations = np.expand_dims(observations, axis=1)
        next_observations = np.expand_dims(next_observations, axis=1)
        actions = np.expand_dims(actions, axis=1)
        rewards = np.expand_dims(rewards, axis=1)
        dones = np.expand_dims(dones, axis=1)
        infos = np.expand_dims(infos, axis=1)
        num_success_true += rewards[-1] == 1
        # aug_observations, aug_next_observations, aug_actions, aug_rewards, aug_dones, aug_infos = augmentation_function.augment(3, observations, next_observations, actions, rewards, dones, infos)

        for j in range(observations.shape[0]):
            aug_obs, aug_next_obs, aug_action, aug_reward, aug_done, aug_info = augmentation_function.augment(
                aug_n, observations[j], next_observations[j], actions[j], rewards[j], dones[j], infos[j])
            num_success += aug_reward[0] == 1
            num_failure += aug_reward[0] < 1
            for k in range(aug_n):
                env.reset()
                env.set_state(np.copy(aug_obs[k, :2]), np.copy(aug_obs[k, 2:]))
                next_obs_true, reward_true, done_true, info_true = env.step(aug_action[k])

                make_assertions(aug_next_obs[k], aug_reward[k], aug_done[k], aug_info[k],
                        next_obs_true, reward_true, done_true, info_true)
    print(num_success, num_failure, num_success_true)
if __name__ == "__main__":

    for env_id, aug_function_classes in AUGMENTATION_FUNCTIONS.items():
        for aug_function_name, aug_function_class in aug_function_classes.items():
            print(env_id, aug_function_name)
            if 'Predator' in env_id and aug_function_name == 'translate_proximal':
                env = gym.make(env_id)
                aug_function = aug_function_class(env=env)
                test_augmentation(env, aug_function)


