import argparse
import os

import gym, my_gym
import numpy as np
from utils import StoreDict, ALGOS

def simulate(env, model, num_actions, seed=0):
    states = []
    actions = []
    rewards = []
    next_states = []
    next_actions =[]
    dones = []

    env.seed(seed)
    np.random.seed(seed)

    action_count = 0
    while action_count < num_actions:

        # s,a,r,s',a',done
        ep_actions = []

        state = env.reset()
        done = False

        step_count = 0
        ret = 0
        while not done:

            action, _ = model.predict(state, deterministic=True)

            states.append(state)
            actions.append(action)
            ep_actions.append(action)

            state, reward, done, _ = env.step(action)
            ret += reward

            rewards.append(reward)
            next_states.append(state)
            dones.append(False)

            step_count += 1

        action_count += step_count
        dones[-1] = True
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

    return states, actions, rewards, next_states, next_actions, dones

if __name__ == "__main__":

    print(os.getcwd())
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str, default=f"InvertedPendulum-v2")
    parser.add_argument("--env-kwargs", type=str, nargs="*", action=StoreDict, default={'init_pos': [0,0]})
    parser.add_argument("--render", type=bool, default=False)
    parser.add_argument("--time-feature", type=bool, default=False)
    parser.add_argument("--algo", type=str, default="td3")
    parser.add_argument("--num-actions", type=int, default=int(100e3))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reward-threshold", type=float, default=-np.inf)
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
    model_path = f'experiments/{env_id}/td3/init_pos_0/run_2/best_model.zip'

    model = algo_class.load(model_path, env=env, custom_objects=custom_objects)
    states, actions, rewards, next_states, next_actions, dones = simulate(
        env=env, model=model, num_actions=args.num_actions)

    save_dir = f'./demonstrations/{env_id}'
    save_path = f'{save_dir}/init_pos_0.npz'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    print(f'Saving to {save_path}')

    np.savez(save_path, states=states, actions=actions, rewards=rewards, dones=dones)