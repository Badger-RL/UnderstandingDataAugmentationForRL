import numpy as np

if __name__ == "__main__":

    n = 10000
    X = np.linspace(-1,1,n).reshape(1,-1)
    Y = np.linspace(-1,1,n).reshape(1,-1)
    D = np.concatenate([X,Y]).T

    for i in range(n):
        x = np.random.uniform