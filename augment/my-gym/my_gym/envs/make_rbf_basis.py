import os.path

import numpy as np
import scipy.spatial
from matplotlib import pyplot as plt

if __name__ == "__main__":


    for rbf_n in [16,32,64,128,256,512,1024]:
        obs_dim = 4

        P = np.random.normal(loc=0, scale=1, size=(rbf_n, obs_dim))
        phi = np.random.uniform(low=-np.pi, high=np.pi, size=(rbf_n,))

        m = 10000
        observations = np.random.uniform(-1, +1, size=(m, obs_dim))
        out = []
        rbfs = []
        for i in range(m):
            out.append((observations[i]))
            rbfs.append(np.sin(P.dot(observations[i])))

        d = scipy.spatial.distance.pdist(out)
        nu = np.average(d)

        rbfs = np.array(rbfs)
        for i in range(4):
            plt.hist(rbfs[:,i], alpha=0.2)
        plt.show()

        print(P)
        print(phi)
        print(nu)

        save_dir = f'rbf_basis/obs_dim_{obs_dim}/n_{rbf_n}'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        np.save(f'{save_dir}/P.npy', P)
        np.save(f'{save_dir}/phi.npy', phi)
        np.save(f'{save_dir}/nu.npy', nu)
