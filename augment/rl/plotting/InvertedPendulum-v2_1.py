from augment.rl.plotting.plot import plot, get_paths

if __name__ == "__main__":

    env_id = 'InvertedPendulum-v2'

    path_dict_all = {}
    path_dict_normal = get_paths(results_dir=f'../results/normal/ddpg/{env_id}', key='no aug', n_trials=5)
    plot(path_dict_normal, f'{env_id}', save_dir=f'figures', save_name=f'{env_id}', n=int(50e3), eval_freq=int(1e3))

    path_dict_all.update(path_dict_normal)
    for n in [2**k for k in range(3, 7)]:
        path_dict_n = get_paths(results_dir=f'../results/n_{n}_sigma_0.1/ddpg/{env_id}', key=f'n={n}, sigma=0.1', n_trials=5)
        path_dict_all.update(path_dict_n)

    plot(path_dict_all, f'{env_id}', save_dir=f'figures', save_name=f'{env_id}', n=int(50e3), eval_freq=int(1e3))


    path_dict_all = {}
    n = 4
    path_dict_normal = get_paths(results_dir=f'../results/normal/ddpg/{env_id}', key='no aug', n_trials=5)
    path_dict_all.update(path_dict_normal)
    for sigma in [0.25, 0.5, 1, 10]:
        path_dict_n = get_paths(results_dir=f'../results/n_{n}_sigma_{sigma}/ddpg/{env_id}', key=f'n={n}, sigma={sigma}', n_trials=5)
        path_dict_all.update(path_dict_n)

    plot(path_dict_all, f'{env_id}', save_dir=f'figures', save_name=f'{env_id}', n=int(50e3), eval_freq=int(1e3))