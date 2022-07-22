import argparse
import os

import gym
import numpy as np
import torch
from stable_baselines3 import DQN

from augment.q_model.auxiliary_models import QModel
from augment.rl.utils import ALGOS, StoreDict

if __name__ == "__main__":

    print(os.getcwd())
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str, default=f"CartPole-v1")
    parser.add_argument("--env-kwargs", type=str, nargs="*", action=StoreDict, default={})
    parser.add_argument("--render", type=bool, default=False)
    parser.add_argument("--algo", type=str, default="dqn")
    parser.add_argument("--run-id", type=int, default=1)
    parser.add_argument("--num-actions", type=int, default=int(10))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument('--random', action='store_true', default=False)
    args = parser.parse_args()

    env_id = args.env_id
    env = gym.make(env_id, **args.env_kwargs)
    env.seed(args.seed)

    # NECESSARY
    custom_objects = {
        "learning_rate": 0.0,
        "lr_schedule": lambda _: 0.0,
        "clip_range": lambda _: 0.0,
    }

    algo_class = ALGOS[args.algo]
    model_path = f'rl/results/{args.algo}/{args.env_id}/run_{args.run_id}/best_model.zip'

    model = DQN.load(model_path, env=env, custom_objects=custom_objects)

    q = model.q_net

    # print(q)

    n = 10
    action_dim = env.observation_space.shape[-1]
    states = np.random.uniform(-1, 1, size=(n, action_dim))
    zeros = np.zeros(shape=(n,1))
    ones = np.ones(shape=(n,1))

    s0 = np.column_stack((states, zeros))
    s1 = np.column_stack((states, ones))

    states_t = torch.from_numpy(states)

    q_model = QModel(state_dim=4, action_dim=2)

    q_model.load_state_dict(torch.load(f'q_model/{env_id}/q_true.pt'))

    qs = q_model(states_t)
    print(qs)
    # print(q_model)
