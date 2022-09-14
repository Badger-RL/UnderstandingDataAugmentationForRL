import argparse
import os.path

import gym, my_gym
import numpy as np
from stable_baselines3 import TD3

from augment.rl.utils import ALGOS


def simulate(env, model, num_episodes, seed=0, render=False):
    env.seed(seed)
    np.random.seed(seed)

    returns = []
    observations = []
    actions = []

    for i in range(num_episodes):
        obs = env.reset()
        done = False

        step_count = 0
        ret = 0
        while not done:

            if model:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = env.action_space.sample() # np.random.uniform(-1, +1, size=env.action_space.shape)
            actions.append(action)
            observations.append((obs))

            obs, reward, done, _ = env.step(action)

            # print(obs)
            if render: env.render()
            ret += reward

            step_count += 1
            # print(reward)
        returns.append(ret)

        print(f'episode {i}: return={ret}',)

    return np.array(actions), np.array(observations)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--algo", help="RL Algorithm", default='dqn', type=str)
    parser.add_argument("--env-id", type=str, default="LunarLander-v2", help="environment ID")
    parser.add_argument("--seed", help="Random generator seed", type=int, default=-1)
    args = parser.parse_args()

    env_kwargs = {'obs_type': 'ram', 'render_mode': 'human'}
    env_kwargs = {'rbf_n': 500}
    env_kwargs = {}
    env = gym.make(args.env_id, **env_kwargs)

    algo_class = ALGOS[args.algo]
    model = algo_class.load(f'rl/results/{args.env_id}/{args.algo}/run_4/best_model.zip', env, custom_objects={})
    actions = simulate(env=env, model=model, num_episodes=10, render=True)

    print(actions)

    save_dir = f'experts/{args.env_id}/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    np.save(f'{save_dir}/actions', actions)