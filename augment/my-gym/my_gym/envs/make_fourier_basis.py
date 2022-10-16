import os.path

import numpy as np
import scipy.spatial
from matplotlib import pyplot as plt

if __name__ == "__main__":

    np.random.seed(0)
    sigma = 0.01

    obs_dim = 4

    for d_fourier in [256, 512, 1024, 2048]:
        for sigma in [0.1, 0.01, 0.001]:
            B = np.random.normal(loc=0, scale=sigma**2, size=(d_fourier//2, obs_dim))
            save_dir = f'fourier_basis/obs_dim_{obs_dim}/d_fourier_{d_fourier}/sigma_{sigma}'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            np.save(f'{save_dir}/B.npy', B)
