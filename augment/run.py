import os

if __name__ == "__main__":


    os.chdir('rl')
    for i in range(10):
        os.system(f'python train.py')