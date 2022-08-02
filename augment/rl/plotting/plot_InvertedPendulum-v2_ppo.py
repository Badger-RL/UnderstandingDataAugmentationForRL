from augment.rl.plotting.plot import plot, get_paths

if __name__ == "__main__":

    env_id = 'InvertedPendulum-v2'

    # path_dict_normal = get_paths(results_dir=f'../2022_07_25/normal/ppo/{env_id}', key='no aug', n_trials=10)
    # path_dict_bc = get_paths(results_dir=f'../2022_07_25/bc_10k_10/ppo/{env_id}', key=f'bc', n_trials=10)
    # path_dict_bc = get_paths(results_dir=f'../2022_07_25/bc_1k_1/ppo/{env_id}', key=f'bc', n_trials=10)
    #
    #
    # path_dict_all = {}
    # path_dict_all.update(path_dict_normal)
    # path_dict_all.update(path_dict_bc)
    #
    # # plot(path_dict_normal, f'{env_id}', save_dir=f'figures', save_name=f'{env_id}', n=int(100e3), eval_freq=int(1e3))
    # plot(path_dict_all, f'{env_id}', save_dir=f'figures', save_name=f'{env_id}', n=int(100e3), eval_freq=int(1e3))

    path_dict_normal = get_paths(results_dir=f'../results/normal/{env_id}/ppo/', key='no aug', n_trials=9)


    path_dict_all = {}
    path_dict_all.update(path_dict_normal)
    plot(path_dict_all, f'{env_id}', save_dir=f'figures', save_name=f'{env_id}', n=int(200e3), eval_freq=int(1e3))