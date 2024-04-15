import gym, my_gym
import numpy as np
from matplotlib import pyplot as plt

from augment.rl.algs.td3 import TD3
from my_gym.envs import LQREnv, LQRGoalEnv

def sim():

    obs_list = []
    num_episodes = 500
    env.seed(0)
    for i in range(num_episodes):
        obs = env.reset()
        done = False

        step_count = 0
        ret = 0
        while not done:

            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            obs_list.append(obs)

        print(f'episode {i}: return={ret}',)

    obs_list = np.array(obs_list)
    plt.scatter(obs_list[:,0], obs_list[:,1])
    plt.axis('equal')
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.xlabel('x position')
    plt.ylabel('y position')
    plt.title('Trajectories of trained agent (no aug)')
    plt.show()

def plot_hist(observations, title):
    fig = plt.figure(figsize=(15,6))
    # observations[:,:,0] = np.arcsin(observations[:,:,0])
    # observations[:,:,1] = np.arcsin(observations[:,:,1])

    plt.suptitle(title)

    for t in range(10):
        T = int(t*10e3)
        # T = 50000
        obs = observations[T:T+10000].reshape(10000,-1)
        x = obs[:, 0]
        y = obs[:, 1]

        plt.subplot(2,5,t+1)
        plt.hist2d(x, y, bins=25, density=False)
        plt.colorbar()
        # plt.axis('equal')
        plt.xlabel('x position')
        plt.ylabel('y position')
        plt.title(f'Steps {T}-{T+10000}')
    fig.tight_layout()
    plt.show()

def plot_hist_2(observations):

    fig = plt.figure(figsize=(15,6))
    # plt.suptitle(f'')

    for t in range(0,10):
        T = int((t+1)*10e3)
        # T = 50000
        obs = observations[:T].reshape(T,-1)
        x = obs[:, 0]
        y = obs[:, 1]

        plt.subplot(2,5,t+1)
        plt.hist2d(x, y, bins=25, density=False)
        plt.colorbar()
        plt.axis('equal')
        plt.xlabel('x position')
        plt.ylabel('y position')
        # plt.title(f'State histogram, steps 0-{T}')
    plt.show()


if __name__ == "__main__":

    env = gym.make('PredatorPrey-v0')

    obs = []
    obs_aug = []
    aug_obs_aug = []
    for run_id in range(2,3):
        path = f'local/results/PredatorPrey-v0/td3/no_aug/run_{run_id}'
        model = TD3.load(f'{path}/best_model.zip', env, custom_objects={})
        model.load_replay_buffer(f'{path}/replay_buffer.pkl')

        aug_path = f'local/results/PredatorPrey-v0/td3/rotate/ratio_1/sched_constant/n_3/run_{run_id}'
        aug_model = TD3.load(f'{aug_path}/best_model.zip', env, custom_objects={})
        aug_model.load_replay_buffer(f'{aug_path}/replay_buffer.pkl')
        aug_model.load_aug_replay_buffer(f'{aug_path}/aug_replay_buffer.pkl')

        obs.append(model.replay_buffer.observations)
        obs_aug.append(aug_model.replay_buffer.observations)
        aug_obs_aug.append(aug_model.aug_replay_buffer.observations)

        plot_hist(model.replay_buffer.observations, title='No Aug Agent: Observed Buffer')
        plot_hist(aug_model.replay_buffer.observations, title='Aug Agent: Observed Buffer')
        plot_hist(aug_model.aug_replay_buffer.observations, title='Aug Agent: Aug Buffer')

    # obs = np.concatenate(obs)
    # obs_aug = np.concatenate(obs_aug)
    # aug_obs_aug = np.concatenate(aug_obs_aug)
    #
    # plot_hist(obs)
    # plot_hist(obs_aug)
    # plot_hist(aug_obs_aug)







