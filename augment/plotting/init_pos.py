from augment.rl.plotting.plot import plot, get_paths, get_paths_auto

if __name__ == "__main__":

    env_ids = ['InvertedPendulum-v2', 'InvertedDoublePendulum-v2']
    algo = 'sac'

    for env_id in env_ids:
        root_dir = f'../condor/init_pos/results/{env_id}/{algo}/'
        path_dict_all = {}
        path_dict_no_aug = get_paths_auto(results_dir=f'../condor/ratio_sched_sweep/results/{env_id}/sac/no_aug',key=fr'$x_0\in$({-0.01},{0.01})')
        path_dict_all.update(path_dict_no_aug)
        for init_pos in [0, 0.2, 0.4, 0.6, 0.8, 1]:
            path_dict_aug = get_paths_auto(results_dir=f'{root_dir}/init_pos_{init_pos}', key=fr'$x_0\in$({-init_pos},{init_pos})')
            path_dict_all.update(path_dict_aug)

        plot(path_dict_all, f'{env_id}: {algo}', save_dir=f'figures', save_name=f'{env_id}', n=int(50e3), eval_freq=int(1e3))

    algo = 'td3'

    for env_id in env_ids:
        root_dir = f'../condor/init_pos/results/{env_id}/{algo}/'
        path_dict_all = {}
        path_dict_no_aug = get_paths_auto(results_dir=f'../condor/ratio_td3/results/{env_id}/td3/no_aug',key=fr'$x_0\in$({-0.01},{0.01})')
        path_dict_all.update(path_dict_no_aug)
        for init_pos in [0,  0.2,  0.4,  0.6, 0.8, 1]:
            path_dict_aug = get_paths_auto(results_dir=f'{root_dir}/init_pos_{init_pos}', key=fr'$x_0\in$({-init_pos},{init_pos})')
            path_dict_all.update(path_dict_aug)

        plot(path_dict_all, f'{env_id}: {algo}', save_dir=f'figures', save_name=f'{env_id}', n=int(50e3), eval_freq=int(1e3))
