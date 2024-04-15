import argparse
import os.path

import gym, my_gym
import numpy as np
from stable_baselines3 import TD3

from augment.rl.utils import ALGOS


def simulate(env, model, num_episodes, seed=0, render=False, flatten=True, verbose=0):
    env.seed(seed)
    np.random.seed(seed)

    observations = []
    next_observations = []
    actions = []
    rewards = []
    returns = []
    dones = []
    infos = []

    for i in range(num_episodes):
        ep_observations, ep_next_observations, ep_actions, ep_rewards, ep_dones, ep_infos,  = [], [], [], [], [], []
        obs = env.reset()
        done = False

        while not done:

            if model:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = env.action_space.sample() # np.random.uniform(-1, +1, size=env.action_space.shape)

            ep_actions.append(action)
            ep_observations.append(obs)

            obs, reward, done, info = env.step(action)

            ep_next_observations.append(obs)
            ep_rewards.append(reward)
            ep_dones.append(done)
            ep_infos.append(info)

            if render: env.render()

        returns.append(sum(ep_rewards))

        if flatten:
            observations.extend(ep_observations)
            next_observations.extend(ep_next_observations)
            actions.extend(ep_actions)
            rewards.extend(ep_rewards)
            dones.extend(ep_dones)
            infos.extend(ep_infos)
        else:
            observations.append(ep_observations)
            next_observations.append(ep_next_observations)
            actions.append(ep_actions)
            rewards.append(ep_rewards)
            dones.append(ep_dones)
            infos.append(ep_infos)
        if verbose:
            print(f'episode {i}: return={returns[-1]}',)

    # return np.array(observations), np.array(actions), np.array(rewards), np.array(infos)
    return np.array(observations), np.array(next_observations), np.array(actions), np.array(rewards), np.array(dones), np.array(infos)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--algo", help="RL Algorithm", default='td3', type=str)
    parser.add_argument("--env-id", type=str, default="PredatorPrey-v0", help="environment ID")
    parser.add_argument("--seed", help="Random generator seed", type=int, default=-1)
    args = parser.parse_args()

    env_kwargs = {'obs_type': 'ram', 'render_mode': 'human'}
    env_kwargs = {'rbf_n': 500}
    env_kwargs = {}
    env = gym.make(args.env_id, **env_kwargs)

    algo_class = ALGOS[args.algo]
    model = algo_class.load(f'rl/baselines/{args.env_id}/{args.algo}/run_0/best_model.zip', env, custom_objects={})
    actions = simulate(env=env, model=None, num_episodes=1, render=False)


    save_dir = f'experts/{args.env_id}/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    np.save(f'{save_dir}/actions', actions)