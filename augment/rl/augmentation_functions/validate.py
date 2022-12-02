import numpy as np

from augment.simulate import simulate


def validate_augmentation(env, aug_func, check_valid):
    observations, next_observations, actions, rewards, dones, infos = simulate(
        model=None, env=env, num_episodes=1, seed=np.random.randint(1,1000000), render=False, flatten=True, verbose=0)

    observations = np.expand_dims(observations, axis=1)
    next_observations = np.expand_dims(next_observations, axis=1)
    actions = np.expand_dims(actions, axis=1)
    rewards = np.expand_dims(rewards, axis=1)
    dones = np.expand_dims(dones, axis=1)
    infos = np.expand_dims(infos, axis=1)

    aug_n = 1

    for j in range(observations.shape[0]):
        aug_obs, aug_next_obs, aug_action, aug_reward, aug_done, aug_info = aug_func.augment(
            aug_n, observations[j], next_observations[j], actions[j], rewards[j], dones[j], infos[j])
        for k in range(aug_n):
            check_valid(env, aug_obs[k], aug_next_obs[k], aug_action[k], aug_reward[k], aug_done[k], aug_info[k])