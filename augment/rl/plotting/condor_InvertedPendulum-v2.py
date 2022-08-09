from augment.rl.plotting.plot import plot, get_paths, get_paths_auto



if __name__ == "__main__":

    env_id = 'InvertedPendulum-v2'

    # path_dict_0 = get_paths(results_dir=f'../experiments/InvertedPendulum-v2/td3/init_pos_0', key='init_pos=0', n_trials=10)
    # path_dict_05 = get_paths_auto(results_dir=f'../experiments/InvertedPendulum-v2/td3/init_pos_0.5', key='init_pos=[-0.5,0.5]')
    # path_dict = {}
    # path_dict.update(path_dict_0)
    # path_dict.update(path_dict_05)
    # plot(path_dict, f'{env_id}', save_dir=f'figures', save_name=f'{env_id}', n=int(100e3), eval_freq=int(1e3))


    path_dict_no_aug = get_paths_auto(results_dir=f'../condor/results/translate/no_aug', key='no aug')
    # plot(path_dict_no_aug, f'{env_id}', save_dir=f'figures', save_name=f'{env_id}', n=int(100e3), eval_freq=int(1e3))
    # for sigma in [0.01, 0.05, 0.1, 0.25, 0.5, 1]:
    #     path_dict_all = {}
    #     path_dict_all.update(path_dict_no_aug)
    #     for ratio in [0.2, 0.4, 0.6, 0.8, 1]:
    #
    #         path_dict_n = get_paths_auto(results_dir=f'../condor/results/translate/ratio_{ratio}/sigma_{sigma}', key=f'ratio={ratio}, sigma={sigma}')
    #         # path_dict_n2 = get_paths_auto(results_dir=f'../condor/trial_2/results/{env_id}/td3/translate/ratio_{ratio}/sigma_{sigma}', key=f'ratio={ratio}, sigma={sigma}')
    #         # path_dict_n.update(path_dict_n2)
    #
    #         path_dict_all.update(path_dict_n)
    #
    #     plot(path_dict_all, f'{env_id}', save_dir=f'figures', save_name=f'{env_id}', n=int(100e3), eval_freq=int(1e3))

    path_dict_all = {}
    for ratio in [0.2, 0.4, 0.6, 0.8, 1]:
        path_dict_all.update(path_dict_no_aug)

        path_dict_n = get_paths_auto(results_dir=f'../condor/results/reflect/ratio_{ratio}', key=f'ratio={ratio}')
        # path_dict_n2 = get_paths_auto(results_dir=f'../condor/trial_2/results/{env_id}/td3/translate/ratio_{ratio}/sigma_{sigma}', key=f'ratio={ratio}, sigma={sigma}')
        # path_dict_n.update(path_dict_n2)

        path_dict_all.update(path_dict_n)

    plot(path_dict_all, f'{env_id}', save_dir=f'figures', save_name=f'{env_id}', n=int(100e3), eval_freq=int(1e3))


    path_dict_0 = get_paths(results_dir=f'../experiments/InvertedPendulum-v2/td3/init_pos_0', key='init_pos=0', n_trials=10)
    path_dict_05 = get_paths_auto(results_dir=f'../experiments/InvertedPendulum-v2/td3/init_pos_0.5', key='init_pos=[-0.5,0.5]')
    path_dict = {}
    path_dict.update(path_dict_0)
    path_dict.update(path_dict_05)
    plot(path_dict, f'{env_id}', save_dir=f'figures', save_name=f'{env_id}', n=int(100e3), eval_freq=int(1e3))
    #     path_dict_n = get_paths(results_dir=f'../experiments/{env_id}/td3/ratio_{aug_ratio}', key=f'r={aug_ratio}', n_trials=10)
    #     path_dict_all.update(path_dict_n)
    #
    # #
    #
    # path_dict_all = {}
    # n = 4
    # path_dict_normal = get_paths(results_dir=f'../results/normal/ddpg/{env_id}', key='no aug', n_trials=5)
    # path_dict_all.update(path_dict_normal)
    # for sigma in [0.25, 0.5, 1, 10]:
    #     path_dict_n = get_paths(results_dir=f'../results/n_{n}_sigma_{sigma}/ddpg/{env_id}', key=f'n={n}, sigma={sigma}', n_trials=5)
    #     path_dict_all.update(path_dict_n)
    #
    # plot(path_dict_all, f'{env_id}', save_dir=f'figures', save_name=f'{env_id}', n=int(50e3), eval_freq=int(1e3))