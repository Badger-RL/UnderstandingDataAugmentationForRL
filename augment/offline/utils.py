import gym
import matplotlib.pyplot as plt
import numpy as np
from d3rlpy.dataset import MDPDataset, Transition
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from d3rlpy.algos import DQN
from d3rlpy.datasets import get_cartpole
from d3rlpy.wrappers.sb3 import SB3Wrapper

def augment(data, n=1000):

    s1, a, r, done = data['states'], data['actions'], data['rewards'],  data['dones']
    s2 = s1.copy()
    s2[:-1,:] = s2[1:, :]
    transitions = []
    for i in range(len(s1)):
        transitions += augment_transition(s1[[i]], a[[i]], r[[i]], s2[[i]], done[[i]], n)

    return transitions


def augment_transition(s1, a, r, s2, done, n=3):

    s1_translate = np.repeat(s1.reshape(1,-1), repeats=n, axis=0)
    lo = -4.8
    hi = +4.8
    # s1_translate[:, 0] = np.linspace(lo, hi, n).T
    s1_translate[:, 0] = np.random.uniform(lo, hi, n).T

    s2_translate = np.repeat(s2.reshape(1,-1), repeats=n, axis=0)
    # s2_translate[:, 0] = np.linspace(lo, hi, n).T
    s1_translate[:, 0] = np.random.uniform(lo, hi, n).T

    s1 = np.concatenate((s1, s1_translate))
    s2 = np.concatenate((s2, s2_translate))
    a  = np.repeat(a, repeats=n+1)
    r  = np.repeat(r, repeats=n+1)

    # s1_reflect = s1_translate.copy()[:,1] * -1
    # s2_reflect = s2_translate.copy()[:,1] * -1
    # a_reflect = ~a

    # s1 = np.concatenate((s1, s1_translate, s1_reflect))
    # s2 = np.concatenate((s2, s2_translate, s2_reflect))
    # a  = np.concatenate((a,  a,            a_reflect))
    # r  = np.concatenate((r,  r,            r))

    x1, theta1 = s1[:, 0], s1[:, 1]
    x2, theta2 = s2[:, 0], s2[:, 1]
    done1 = (x1 < -4.8) | (x1 > 4.8) | (theta1 < -0.418) | (theta1 > 0.418)
    done2 = (x2 < -4.8) | (x2 > 4.8) | (theta2 < -0.418) | (theta2 > 0.418)
    done = done1 & done2

    transitions = []
    for i in range(n+1):
        transitions.append(
            Transition(observation_shape=(4,), action_size=2,
                       observation=s1[i], action=a[i], reward=r[i], next_observation=s2[i], terminal=done[i]))

    return transitions

def translate(s, n=10):
    s_translate = np.repeat(s.reshape(1,-1), repeats=n, axis=0)
    s_translate[:, 0] = np.linspace(-4.9, 4.9, n).T
    return s_translate

def reflect_and_translate(s, n=10):
    s_reflect = translate(s, n)
    s_reflect[:,1:] *= -1
    return s_reflect

def augment_transition_old(s, a, r, sp, n=10):

    s_translate = translate(s, n)
    s_reflect = reflect_and_translate(s, n)

    s = np.concatenate((s_translate, s_reflect))

    sp_translate = translate(sp, n)
    sp_reflect = reflect_and_translate(sp, n)

    sp = np.concatenate((sp_translate, sp_reflect))


    x, theta = s[:,0], s[:,1]
    done = (x < -4.8) | (x > 4.8) |(theta < -0.418) | (theta > 0.418)

    a = np.repeat(a, repeats=2*n)
    r = np.ones(2*n)

    transitions = []
    for i in range(2*n):
        transitions.append(
            Transition(observation_shape=(4,), action_size=2,
                       observation=s[i], action=a[i], reward=r[i], next_observation=sp[i], terminal=done[i]))

    return transitions
    # return MDPDataset(observations=s, actions=a, rewards=r, terminals=done, discrete_action=True)

def augment_data(transitions, n=1000):

    for tau in transitions:
        s_translate = translate(s, n)
    s_reflect = reflect_and_translate(s, n)

    s = np.concatenate((s_translate, s_reflect))

    sp_translate = translate(sp, n)
    sp_reflect = reflect_and_translate(sp, n)

    sp = np.concatenate((sp_translate, sp_reflect))


    x, theta = s[:,0], s[:,1]
    done = (x < -4.8) | (x > 4.8) |(theta < -0.418) | (theta > 0.418)

    a = np.repeat(a, repeats=2*n)
    r = np.ones(2*n)

    transitions = []
    for i in range(2*n):
        transitions.append(
            Transition(observation_shape=(4,), action_size=2,
                       observation=s[i], action=a[i], reward=r[i], next_observation=sp[i], terminal=done[i]))

    return transitions

def dataset_augmented():
    env = gym.make('CartPole-v1')
    dataset = []
    for i in range(10):
        env.reset()

        s0 = np.random.uniform(-0.05, 0.05, size=(4,))
        # s0 = np.array([0, 0.1, 0.4181, 0.1])
        env.state = s0
        a0 = 0
        s1, r0, done, _ = env.step(a0)
        dataset1 = augment_transition(s0, 0, r0, s1, n=100)

        s0 = np.random.uniform(-0.05, 0.05, size=(4,))
        # s0 = np.array([0, 0.1, 0.4181, 0.1])
        env.state = s0
        a0 = 1
        s1, r0, done, _ = env.step(a0)
        dataset2 = augment_transition(s0, 1, r0, s1, n=100)
        dataset += dataset1 + dataset2

    return dataset

def load_dataset(path):

    data = np.load(path)
    actions = data['actions']
    states = data['states']
    rewards = data['rewards']
    dones = data['dones']

    next_states = np.copy(states)
    next_states[:-1, :] = next_states[1:, :]

    transitions = []

    for i in range(len(data)):
        transitions.append(
            Transition(observation_shape=(4,), action_size=2,
                       observation=states[i], action=actions[i], reward=rewards[i], next_observation=next_states[i], terminal=dones[i]))

    return transitions
def dataset_random(n=1000):

    env = gym.make('CartPole-v1')

    transitions = []

    for i in range(n):
        env.reset()
        s0 = np.random.uniform(low=[-1, -0.418, -1, -1], high=[1, 0.418, 1, 1], size=(n, 4))
        env.state = s0
        a0 = 0 if np.random.uniform(-1,1) < 0.5 else 1

        s1, r, done, info = env.step(a0)

        transitions.append(
            Transition(observation_shape=(4,), action_size=2,
                       observation=s0, action=a0, reward=r, next_observation=s1, terminal=done))

    return transitions




