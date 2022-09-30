import copy
import seaborn

import gym, my_gym
import numpy as np
import scipy.special
from matplotlib import pyplot as plt, colors

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

def plot_hist(obs, log=False):


    my_cmap = copy.copy(plt.cm.get_cmap())  # copy the default cmap
    my_cmap.set_bad(my_cmap.colors[0])

    x = obs[:, 0]
    y = obs[:, 1]

    norm = None
    if log:
        norm = colors.LogNorm()
    plt.hist2d(x, y, bins=25, density=False, norm=norm, cmap=my_cmap)
    plt.colorbar()
    plt.xlabel('x position')
    plt.ylabel('y position')


if __name__ == "__main__":

    seaborn.set_theme()
    env = gym.make('PredatorPrey-v0')




    plt.figure(figsize=(30,6))
    i=1
    for r in [0.2,0.4,0.6,0.8, 1]:
        plt.subplot(1,5,i)
        i+=1

        obs_opt = np.load(f'../stability_at_convergence/data/PredatorPrey-v0/opt/0_{r}/observations.npy').reshape(-1, 4)
        obs_opt = obs_opt[:, :2]
        hist_opt, _ = np.histogramdd(obs_opt, bins=50, density=True)
        eps = 1e-3

        for aug in ['no_aug','rotate', 'translate', 'translate 0.1']:
            obs = []
            obs_aug = []
            aug_obs_aug = []

            kl_both_all = []
            kl_aug_all = []
            kl_obs_all = []
            no_aug = False
            if '0.1' in aug and r==1:
                continue
            for run_id in range(10):
                if aug == 'no_aug':
                    no_aug = True
                    path = f'../../condor/pp_kl_2/results/PredatorPrey-v0/td3/r_{r}/no_aug/run_{run_id}'
                elif '0.1' in aug:
                    path = f'../../condor/pp_kl/results/PredatorPrey-v0/td3/r_{r}/translate/n_3/run_{run_id}'
                else:
                    path = f'../../condor/pp_kl_2/results/PredatorPrey-v0/td3/r_{r}/{aug}/n_3/run_{run_id}'
                print(path)
                model = TD3(policy='MlpPolicy', env=env)
                try:
                    model.load_replay_buffer(f'{path}/replay_buffer.pkl')
                    if not no_aug:
                        model.load_aug_replay_buffer(f'{path}/aug_replay_buffer.pkl')
                except:
                    print('Skipping...')
                    continue

                # obs.append(model.replay_buffer.observations)
                # obs_aug.append(model.aug_replay_buffer.observations)


                obs = model.replay_buffer.observations.reshape(-1, 4)
                obs = obs[:, :2]

                if not no_aug:
                    obs_aug = model.aug_replay_buffer.observations.reshape(-1, 4)
                    obs_aug = obs_aug[:, :2]

                kl_both = []
                kl_obs = []
                kl_aug = []
                for j in range(20):
                    delta = 10000
                    start = j*delta
                    # start = 0
                    end = (j+1)*delta
                    # hist_no_aug, _ = np.histogramdd(obs, bins=50, density=True)
                    # hist_both, _ = np.histogramdd(obs_both[start:end], bins=50, density=True)
                    hist_obs, _ = np.histogramdd(obs[start:end], bins=50, density=True, range=[(-1,1), (-1,1)])
                    hist_obs = (hist_obs+eps)/(1+eps)
                    kl_matrix = scipy.special.kl_div(hist_opt, hist_obs)
                    inf_mask = np.isinf(kl_matrix)
                    kl_matrix[inf_mask] = 0
                    kl_obs.append(kl_matrix.sum())

                    if not no_aug:
                        hist_aug, _ = np.histogramdd(obs_aug[3*start:3*end], bins=50, density=True, range=[(-1,1), (-1,1)])
                        hist_aug = (hist_aug + eps) / (1 + eps)

                        hist_both = (hist_obs + hist_aug)/2

                        kl_matrix = scipy.special.kl_div(hist_opt, hist_both)
                        inf_mask = np.isinf(kl_matrix)
                        kl_matrix[inf_mask] = 0
                        kl_both.append(kl_matrix.sum())

                        kl_matrix = scipy.special.kl_div(hist_opt, hist_aug)
                        inf_mask = np.isinf(kl_matrix)
                        kl_matrix[inf_mask] = 0
                        kl_aug.append(kl_matrix.sum())

                    # print(kl)

                kl_both_all.append(kl_both)
                kl_aug_all.append(kl_aug)
                kl_obs_all.append(kl_obs)

            kl_both_all = np.array(kl_both_all)
            kl_aug_all = np.array(kl_aug_all)
            kl_obs_all = np.array(kl_obs_all)

            t = np.arange(10000,200e3+1, 10000)

            # linsetyle = '-' if aug=='rotate' else '-.'
            # linsetyle = '-' if aug=='rotate' else '-.'
            c = seaborn.color_palette()[0]
            if aug=='rotate':
                c = seaborn.color_palette()[1]
            if aug=='translate':
                c = seaborn.color_palette()[2]
            elif aug=='translate 0.1':
                c = seaborn.color_palette()[3]


            #
            # avg = np.average(kl_obs_all, axis=0)
            # std = np.std(kl_obs_all, axis=0)
            # linestyle = '--'
            # plt.plot(t, avg, label=f'{aug} observed only',  linestyle=linestyle, color=c)
            # plt.fill_between(t, avg-std, avg+std, alpha=0.2, color=c)

            if not no_aug:
                avg = np.average(kl_both_all, axis=0)
                std = np.std(kl_both_all, axis=0)
                linestyle = '-'
                plt.plot(t, avg, label=f'{aug} combined',  linestyle=linestyle, color=c)
                plt.fill_between(t, avg-std, avg+std, alpha=0.2, color=c)

                # avg = np.average(kl_aug_all, axis=0)
                # std = np.std(kl_aug_all, axis=0)
                # linestyle = ':'
                # plt.plot(t, avg, label=f'{aug} augmented only',  linestyle=linestyle, color=c)
                # plt.fill_between(t, avg-std, avg+std, alpha=0.2, color=c)



            plt.xlabel('Age (timestep) of oldest transition in replay buffer slice')
            plt.ylabel('$D_{KL}(d_{\pi^*}||\\tilde\pi_t)$ ')
            plt.title(fr'$d = {r}$')
            plt.legend()

    plt.suptitle('KL Divergence Between ' r'$d_{\pi^*}$' ' and Replay Buffer(s)')
    plt.tight_layout()
    plt.show()

    # obs = np.concatenate(obs).reshape(-1, 4)
    # obs_aug = np.concatenate(obs_aug).reshape(-1, 4)

    # fig = plt.figure(figsize=(15,6))
    # plt.suptitle('Replay (Marginal) State Distribution')

    # plot_hist(obs, '')
    # plt.show()
    #
    # plot_hist(obs_aug, '')
    # plt.show()








