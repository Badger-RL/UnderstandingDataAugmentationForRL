import argparse

import gym, my_gym
import numpy as np

# from augment.rl.algs.td3 import TD3
from augment.rl.algs.td3 import TD3
from augment.rl.utils import ALGOS

def simulate(env, model, num_episodes, seed=0):
    env.seed(seed)
    np.random.seed(seed)

    returns = []
    inits = []
    for i in range(num_episodes):

        # s,a,r,s',a',done
        ep_actions = []

        obs = env.reset()
        init_pos = obs[0]
        inits.append(init_pos)
        done = False

        step_count = 0
        ret = 0
        while not done:

            if model:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = np.random.uniform(-1, +1, size=env.action_space.shape)

            ep_actions.append(action)

            obs, reward, done, _ = env.step(action)
            env.render()
            ret += reward

            step_count += 1

        returns.append(ret)

        print(f'episode {i}: init_pos={init_pos}, return={ret}',)
    return inits, returns

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--algo", help="RL Algorithm", default="td3", type=str, required=False, choices=list(ALGOS.keys()))
    parser.add_argument("--env-id", type=str, default="InvertedPendulum-v2", help="environment ID")
    parser.add_argument("--seed", help="Random generator seed", type=int, default=-1)
    args = parser.parse_args()

    env = gym.make(args.env_id)
    model = TD3.load('rl/results/InvertedPendulum-v2/td3/run_158/best_model.zip', env, custom_objects={})
    simulate(env=env, model=model, num_episodes=100)