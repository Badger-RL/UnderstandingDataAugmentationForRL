import numpy as np
from torch.utils.data import Dataset

from augment.rl.augmentation_functions import PendulumTranslateUniform


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

def load_dataset(data_path, n=None, slice=0):

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
        start = int(slice)
        end = int(slice+n)

        actions = actions[start:end,:]
        states = states[start:end,:]
        rewards = rewards[start:end]
        dones = dones[start:end]


    n = 100
    delta = np.random.uniform(low=-1, high=+1, size=(n * len(actions),))
    aug = PendulumTranslateUniform()
    states1, next_states1, actions1, rewards1, dones1, infos1 = aug._deepcopy_transition(n, obs=states, next_obs=states, action=actions, reward=rewards, done=dones, infos=[{}])
    states1[:,0] = delta
    states = np.concatenate((states, states1))
    actions = np.concatenate((actions, actions1))
    rewards = np.concatenate((rewards, rewards1))
    dones = np.concatenate((dones, dones1))
    # return TransitionDataset(states, actions, rewards, next_states, next_actions, dones)
    return TransitionDataset(states, actions, rewards, dones)