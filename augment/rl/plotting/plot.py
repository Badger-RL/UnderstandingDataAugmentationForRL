import os

import numpy as np
import seaborn as sns
# from plotting.plot import plot
from matplotlib import pyplot as plt

def get_line_styles(name):

    parts = name.split(' ')
    if len(parts) > 1:
        name = parts[0]
        q = int(parts[1])

    colors = sns.color_palette(n_colors=10)

    if name == 'native':
        linestyle = '-'
        linewidth=3
        color=colors[0]
    elif name.startswith('AE'):
        linestyle = '-'
        linewidth=2
        color=colors[q+1]
    else:
        linestyle = '-.'
        linewidth=2
        color=colors[q+1]

    style_dict = {
        'linestyle': linestyle,
        'linewidth': linewidth,
        'color': color,
    }

    return style_dict

def load_data(path, n, eval_freq):
    try:
        with np.load(path) as data:
            t = data['timesteps'][:n // eval_freq]
            r = data['results'][:n // eval_freq]
            avg = np.average(r, axis=1)
    except:
        print(f'{path} not found')
        return None, None

    return t, avg

def plot(save_dict, title, save_dir, save_name, n=int(2e6), eval_freq=int(1e3)):
    i=0
    fig = plt.figure(figsize=(7,7))

    for agent, paths in save_dict.items():
        avgs = []
        for path in paths:
            t, avg = load_data(path, n, eval_freq)
            if avg is not None:
                avgs.append(avg)

        if len(avgs) == 0: continue
        elif len(avgs) == 1:
            avg_of_avgs = avg
            q05 = np.zeros_like(avg)
            q95 = np.zeros_like(avg)

        else:
            avg_of_avgs = np.average(avgs, axis=0)
            std = np.std(avgs, axis=0)
            N = len(avgs)
            ci = 1.96 * std / np.sqrt(N)
            q05 = avg_of_avgs + ci
            q95 = avg_of_avgs - ci

        # style_dict = get_line_styles(agent)

        # plt.plot(t, avg_of_avgs, label=agent, **style_dict)
        # plt.fill_between(t, q05, q95, alpha=0.2, color=style_dict['color'])
        if t is None:
            t = np.linspace(eval_freq, n+1, eval_freq)
        plt.plot(t, avg_of_avgs, label=agent)
        plt.fill_between(t, q05, q95, alpha=0.2)
        plt.title(title, fontsize=16)
        plt.xlabel('Timestep', fontsize=16)
        plt.ylabel('Return', fontsize=16)

        i+=1

    plt.legend(loc='lower right')
    plt.show()

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    fig.savefig(f'{save_dir}/{save_name}')

def get_paths(results_dir, key, n_trials=10):

    path_dict = {}
    path_dict[key] = []
    for j in range(n_trials):
        path_dict[key].append(f'./{results_dir}/run_{j+1}/evaluations.npz')
    return path_dict

def get_paths_auto(results_dir, key):

    path_dict = {}
    path_dict[key] = []
    for dirpath, dirnames, filenames in os.walk(results_dir):
        path_dict[key].append(f'./{dirpath}/evaluations.npz')
    return path_dict