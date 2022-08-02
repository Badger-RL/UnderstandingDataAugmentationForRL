import os

if __name__ == "__main__":


    os.chdir('rl')

    # for i in range(10):
    #     os.system(f'python train.py --run-id {i} --env InvertedPendulum-v2 --algo ppo --eval-freq 1000 -n 200000'
    #               f' -f results/normal')

    for i in range(10):
        os.system(f'python train.py --run-id {i} --env InvertedPendulum-v2 --algo ppo --eval-freq 1000 -n 200000'
                  f' -aug-function cartpole_translate -aug-n 100'
                  f' -f results/bc_every_update')