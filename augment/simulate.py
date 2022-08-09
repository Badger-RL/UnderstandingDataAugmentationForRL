import gym, my_gym
import matplotlib.pyplot as plt
import numpy as np
import seaborn
import torch

from augment.rl.utils import ALGOS


def simulate(env, model, num_episodes, seed=0):
    env.seed(seed)
    np.random.seed(seed)

    rets = []
    inits = []
    for i in range(num_episodes):

        # s,a,r,s',a',done
        ep_actions = []

        state = env.reset()
        init_pos = state[0]
        inits.append(init_pos)
        done = False

        step_count = 0
        ret = 0
        while not done:

            action, _ = model.predict(state, deterministic=True)

            ep_actions.append(action)

            state, reward, done, _ = env.step(action)
            # env.render()
            ret += reward

            step_count += 1

        ep_return = np.sum(ret)
        rets.append(ret)
        print(f'episode {i}: init_pos={init_pos}, return={ep_return}',)
    return inits, rets
