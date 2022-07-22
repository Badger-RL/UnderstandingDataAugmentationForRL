import argparse
import os
import sys

import gym
import numpy as np
from augment.rl.utils import StoreDict, ALGOS
# from utils import ALGOS

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)


def heuristic(obs):
    theta, w = obs[2:4]
    if abs(theta) < 0.03:
        return 0 if w < 0 else 1
    else:
        return 0 if theta < 0 else 1

def simulate(env, model, num_actions, random=False, seed=0):
    states = []
    actions = []
    rewards = []
    next_states = []
    next_actions =[]
    dones = []

    discrete = len(env.action_space.shape) == 0
    if discrete:
        action_dim = env.action_space.n
    else:
        action_dim = env.action_space.shape[-1]

    env.seed(seed)
    np.random.seed(seed)

    action_count = 0
    while action_count < num_actions:

        # s,a,r,s',a',done
        ep_actions = []

        state = env.reset()
        # state = np.random.uniform(low=[-4.8, -1, -0.418, -1], high=[4.8, 1, 0.418, 1], size=(4,))
        done = False

        step_count = 0
        ret = 0
        while not done:


            # if random:
            #     action = np.random.uniform(-1, 1, size=(action_dim,))
            # else:
            #     action, _ = model.predict(state, deterministic=True)
            action = np.array(heuristic(obs=state))

            states.append(state)
            actions.append(action)
            ep_actions.append(action)

            state, reward, done, _ = env.step(action)
            ret += reward

            rewards.append(reward)
            next_states.append(state)
            dones.append(done)

            step_count += 1

        action_count += step_count
        # dones[-1] = True
        next_actions.extend(ep_actions[1:])
        next_actions.append(ep_actions[-1])

        ep_return = np.sum(ret)
        print(f'action_count = {action_count}, return = {ep_return}', step_count)

    states = np.array(states)
    actions = np.array(actions)
    rewards = np.array(rewards)
    next_states = np.array(next_states)
    next_actions = np.array(next_actions)
    dones = np.array(dones)

    # if discrete:
    #     actions_one_hot = np.zeros((actions.size, actions.max() + 1))
    #     actions_one_hot[np.arange(actions.size), actions] = 1
    #     actions = actions_one_hot

    return states, actions, rewards, next_states, next_actions, dones

if __name__ == "__main__":

    print(os.getcwd())
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str, default=f"CartPole-v1")
    parser.add_argument("--env-kwargs", type=str, nargs="*", action=StoreDict, default={})
    parser.add_argument("--render", type=bool, default=False)
    parser.add_argument("--algo", type=str, default="ppo")
    parser.add_argument("--run-id", type=int, default=1)
    parser.add_argument("--num-actions", type=int, default=int(2500))
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

    model = algo_class.load(model_path, env=env, custom_objects=custom_objects)
    states, actions, rewards, next_states, next_actions, dones = simulate(seed=1,
        env=env, model=model, num_actions=args.num_actions, random=args.random)

    print(states)
    print(actions)
    print(rewards)

    save_dir = f'./data/{env_id}'
    if args.random:
        save_path = f'{save_dir}/random.npz'
    else:
        save_path = f'{save_dir}/trained.npz'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    print(f'Saving to {save_dir}')

    np.savez(save_path, states=states, actions=actions, rewards=rewards, dones=dones)