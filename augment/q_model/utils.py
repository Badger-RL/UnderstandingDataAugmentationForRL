import argparse
import json
import os

import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import Dataset

from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3

ALGOS = {
    "a2c": A2C,
    "ddpg": DDPG,
    "dqn": DQN,
    "ppo": PPO,
    "sac": SAC,
    "td3": TD3,
}

class StoreDict(argparse.Action):
    """
    Custom argparse action for storing dict.

    In: args1:0.0 args2:"dict(a=1)"
    Out: {'args1': 0.0, arg2: dict(a=1)}
    """

    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        self._nargs = nargs
        super(StoreDict, self).__init__(option_strings, dest, nargs=nargs, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        arg_dict = {}
        for arguments in values:
            key = arguments.split(":")[0]
            value = ":".join(arguments.split(":")[1:])
            # Evaluate the string as python code
            arg_dict[key] = eval(value)
        setattr(namespace, self.dest, arg_dict)

class TransitionDataset(Dataset):
    def __init__(self, states, actions, rewards, dones, normalize_rewards=True):
        self.states = states[:,:]
        self.robot_states = np.copy(self.states[:,:])
        self.actions = actions
        self.rewards = rewards
        # self.next_states = next_states
        # self.next_actions = next_actions
        # if normalize_rewards:
        #     min_reward = np.min(self.rewards)
        #     max_reward = np.max(self.rewards)
        #     self.rewards = (self.rewards-min_reward)/(max_reward-min_reward)
        self.robot_state_dim = self.robot_states.shape[-1]
        self.state_dim = self.states.shape[-1]
        self.action_dim = self.actions.shape[-1]

        self.dones = dones

        self.next_states = np.copy(self.states)
        self.next_states[:-1, :] = self.next_states[1:, :]
        self.next_robot_states = np.copy(self.robot_states)
        self.next_robot_states[:-1, :] = self.next_robot_states[1:, :]

        self.next_actions = np.copy(actions)
        self.next_actions[:-1, :] = self.next_actions[1:, :]
        # self.next_actions[self.dones, :] = np.empty(shape=self.action_dim)

    def __getitem__(self, index):
        state = self.states[index]
        action = self.actions[index]
        reward = self.rewards[index]
        next_state = self.next_states[index]
        next_action = self.next_actions[index]
        done = self.dones[index]

        robot_state = self.robot_states[index]
        next_robot_state = self.next_robot_states[index]

        return {'state': state, 'action': action, 'reward': reward, 'next_state': next_state, 'next_action': next_action, 'done': done,
                'robot_state': robot_state, 'next_robot_state': next_robot_state}

    def __len__(self):
        return len(self.actions)

def load_dataset(data_path, n=None):

    data = np.load(data_path)
    actions = data['actions']
    states = data['states']
    rewards = data['rewards']
    # next_states = data['next_states']
    # next_actions = data['next_actions']
    dones = data['dones']

    rewards = rewards.reshape(-1)
    dones = dones.reshape(-1)

    if n is not None:
        actions = actions[:n,:]
        states = states[:n,:]
        rewards = rewards[:n]
        dones = dones[:n]

    # return TransitionDataset(states, actions, rewards, next_states, next_actions, dones)
    return TransitionDataset(states, actions, rewards, dones)

def plot_loss(logs, save_dir=None, filename=None):
    for key, val in logs.items():
        plt.plot(val, label=key)
    plt.title('Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.yscale('log')

    if save_dir is not None and not os.path.exists(save_dir):
        os.makedirs(save_dir)
        plt.savefig(f'{save_dir}/{filename}')

    plt.show()

if __name__ == "__main__":

    with open('tmp3/Ant-v3/logs.json') as f:
        logs = json.load(f)
        plot_loss(logs)