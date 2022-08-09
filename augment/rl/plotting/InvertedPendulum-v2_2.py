from augment.rl.plotting.plot import plot, get_paths, get_paths_auto

if __name__ == "__main__":

    env_id = 'InvertedPendulum-v2'

    path_dict_all = {}
    # path_dict_normal = get_paths(results_dir=f'../results/no_aug/{env_id}/td3/', key='no aug', n_trials=15)
    path_dict_normal = get_paths(results_dir=f'../experiments/{env_id}/td3/no_aug', key='no aug', n_trials=10)
    path_dict_normal = get_paths_auto(results_dir=f'../condor/results/translate/no_aug', key='no aug')
    path_dict_only_aug = get_paths(results_dir=f'../experiments/{env_id}/td3/only_aug', key='only aug, sigma=0.1', n_trials=10)

    # plot(path_dict_normal, f'{env_id}', save_dir=f'figures', save_name=f'{env_id}', n=int(50e3), eval_freq=int(1e3))
    #
    path_dict_all.update(path_dict_normal)
    path_dict_all.update((path_dict_only_aug))
    # for aug_ratio in [0.25, 0.5, 0.75, 1]:
    #     # path_dict_n = get_paths(results_dir=f'../results/ratio_{aug_ratio}_n_{1}/{env_id}/td3', key=f'r={aug_ratio}, n={1}, sigma=0.1', n_trials=10)
    #     # path_dict_all.update(path_dict_n)
    #     path_dict_n = get_paths(results_dir=f'../experiments/{env_id}/td3/ratio_{aug_ratio}', key=f'r={aug_ratio}', n_trials=10)
    #     path_dict_all.update(path_dict_n)

    plot(path_dict_all, f'{env_id}', save_dir=f'figures', save_name=f'{env_id}', n=int(50e3), eval_freq=int(1e3))
    #
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