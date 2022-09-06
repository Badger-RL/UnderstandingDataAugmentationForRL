import os

if __name__ == "__main__":


    os.chdir('rl')
    env_id = "CartPole-v1"

    for i in range(1,10):
        os.system(f'python train.py --run-id {i} --env {env_id} --algo dqn --eval-freq 1000 -n 50000'
                  f' -f ../data -exp no_aug'
                  )

    # for i in range(1,21):
    #     os.system(f'python train.py --run-id {i} --env {env_id} --algo dqn --eval-freq 1000 -n 25000'
    #               f' -f ../data -exp translate_uniform/ratio_1/constant/n_1'
    #               f' --aug-function translate_uniform --aug-n 1'
    #               )
        # os.system(f'python train.py --run-id {i} --env InvertedPendulum-v2 --algo sac --eval-freq 1000 -n 30000'
        #           # f' --env-kwargs init_pos:-1,1'
        #           f' --aug-function translate_uniform --aug-schedule \"constant\" --aug-constraint 0'
        #           f' -f ../data -exp translate_uniform_0')

    # for aug_n in [1]:
    #     for aug_ratio in [1]:
    #         for i in range(5):
    #             os.system(f'python train.py --env InvertedPendulum-v2 --algo td3 --eval-freq 1000 -n 50000'
    #                       f' -f results/ratio_{aug_ratio}_n_{aug_n}'
    #                       f' -aug-function cartpole_translate'
    #                       f' -aug-ratio {aug_ratio}')
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