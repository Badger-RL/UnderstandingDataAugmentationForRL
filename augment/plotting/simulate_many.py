import gym
import numpy as np
import seaborn
import torch
from matplotlib import pyplot as plt

from augment.rl.utils import ALGOS
from augment.simulate import simulate

if __name__ == "__main__":

    seaborn.set_theme()
    env_id = 'InvertedPendulum-v2'
    env_kwargs = {'init_pos': [-0.5,0.5]}
    seed = 0
    algo = 'td3'

    env = gym.make(env_id, **env_kwargs)
    env.seed(seed)

    algo_class = ALGOS[algo]
    custom_objects = {}

    inits_bc = []
    rets_bc = []
    for i in range(5):
        policy_path = f'./tmp/InvertedPendulum-v2/policy_{i}.pt'
        policy = torch.load(policy_path)
        model = algo_class(policy='MlpPolicy', env=env)
        model.policy = policy
        inits, rets = simulate(env=env, model=model, num_episodes=100, )
        print(rets)
        inits_bc.extend(inits)
        rets_bc.extend(rets)
    model_path = f'./rl/experiments/InvertedPendulum-v2/td3/init_pos_0/run_2/best_model.zip'
    # model = TD3(policy='MlpPolicy', env=env)
    model = algo_class.load(model_path, custom_objects=custom_objects, env=env)
    inits, rets = simulate(env=env, model=model, num_episodes=1000,)
    ci = np.std(rets)/np.sqrt(1000)*1.96
    print(np.average(rets), ci)

    plt.scatter(inits_bc, rets_bc, label='BC policy')
    plt.scatter(inits, rets, label='slice policy')
    plt.xlim(-1, +1)
    plt.xlabel('initial x position')
    plt.ylabel('return')
    plt.title(f'{env_id}: Return vs Initial Position')
    plt.legend()

    # plt.bar(x=['Slice Policy', 'BC Policy'], height=[179, 100], yerr=[305/np.sqrt(1000)*1.96, 20])
    plt.show()