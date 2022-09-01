from augment.rl.plotting.plot import plot, get_paths, get_paths_auto

if __name__ == "__main__":


    root_dir = f'../data/CartPole-v1-good/dqn'
    path_dict_no_aug = get_paths_auto(results_dir=f'{root_dir}/no_aug',key='no aug')
    print(path_dict_no_aug)

    path_dict_all = {}
    path_dict_all.update(path_dict_no_aug)

    for sched in ['constant', 'exponential']:
        path_dict_aug = get_paths_auto(results_dir=f'{root_dir}/translate_uniform/ratio_1/{sched}/n_1/', key=fr'sched={sched}')
        path_dict_all.update(path_dict_aug)

    plot(path_dict_all, f'CartPole-v1: dqn', save_dir=f'figures', save_name=f'CartPole-v1', n=int(50e3), eval_freq=int(1e3))



    root_dir = f'../data/CartPole-v1/dqn'
    path_dict_no_aug = get_paths(results_dir=f'{root_dir}/no_aug',key='no aug', n_trials=9)
    print(path_dict_no_aug)

    path_dict_all = {}
    path_dict_all.update(path_dict_no_aug)

    for sched in ['constant', 'exponential']:
        path_dict_aug = get_paths(results_dir=f'{root_dir}/translate_uniform/ratio_1/{sched}/n_1/', key=fr'sched={sched}', n_trials=9)
        path_dict_all.update(path_dict_aug)

    plot(path_dict_all, f'CartPole-v1: dqn', save_dir=f'figures', save_name=f'CartPole-v1', n=int(50e3), eval_freq=int(1e3))