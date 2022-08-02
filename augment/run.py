import os

if __name__ == "__main__":


    os.chdir('rl')

    # for i in range(5):
    #     os.system(f'python train.py --env InvertedPendulum-v2 --algo td3 --eval-freq 1000 -n 50000'
    #               f' -f results/no_aug')

    for aug_n in [1]:
        for aug_ratio in [1]:
            for i in range(5):
                os.system(f'python train.py --env InvertedPendulum-v2 --algo td3 --eval-freq 1000 -n 50000'
                          f' -f results/ratio_{aug_ratio}_n_{aug_n}'
                          f' -aug-function cartpole_translate'
                          f' -aug-ratio {aug_ratio}')
    #
    # for n_aug in [2**k for k in range(10)]:
    #     for i in range(5):
    #         os.system(f'python train.py --env InvertedPendulum-v2 --algo ddpg --eval-freq 1000 -n 50000'
    #                   f' -f results/n_{n_aug}_sigma_0.1'
    #                   f' -aug-kwargs n:{n_aug} sigma:0.1')
    #

    # n_aug = 4
    # for sigma in [0.01, 0.05, 0.25, 0.5, 1, 10]:
    #     for i in range(5):
    #         os.system(f'python train.py --env InvertedPendulum-v2 --algo ddpg --eval-freq 1000 -n 50000'
    #                   f' -f results/n_{n_aug}_sigma_{sigma}'
    #                   f' -aug-kwargs n:{n_aug} sigma:{sigma}')

    # for i in range(10):
    #     os.system(f'python train.py --env InvertedPendulum-v2 --algo ppo --eval-freq 1000 -n 100000'
    #               f' -f results/normal')
    #
    # for i in range(10):
    #     os.system(f'python train.py --env InvertedPendulum-v2 --algo ppo --eval-freq 1000 -n 100000'
    #               f' -f results/bc_10k_20')