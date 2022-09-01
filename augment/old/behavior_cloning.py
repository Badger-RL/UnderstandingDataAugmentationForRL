import argparse
import json
import os
from collections import defaultdict

import gym
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from augment.q_model.utils import plot_loss
from augment.rl.algs.td3 import TD3
from augment.rl.augmentation_functions import PendulumTranslateUniform
from augment.util import load_dataset


def mkdirs(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

device = 'cpu'
class TrainerBase:
    def __init__(self, model, lr=1e-4, verbose=1):
        self.model = model.to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.logs = defaultdict(lambda: [])
        self.verbose = verbose
        self.epoch_num = 0
        self.max_grad = 1
        self.lr = lr
        # self.scheduler = torch.optim.lr_scheduler.ConstantLR(self.optimizer)

        # Being dynamic is harder than simply accounting for all possiblities (for small n)

    def train(self, dataset, save_dir='tmp', save_model_name='policy', batch_size=64, epochs=100):

        # dataset = numpy array [s, a, s']
        dl = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        min_loss = np.inf

        for epoch in range(epochs):
            epoch_losses = defaultdict(lambda: [])

            for batch in dl:
                loss_dict = self.loss(batch)
                batch_total_loss = loss_dict['total_loss']
                self._step_optimizer(batch_total_loss)

                for loss_name, loss in loss_dict.items():
                    epoch_losses[loss_name].append(loss.item())

            # log epoch losses
            epoch_loss_avgs = {}
            for loss_name, loss in loss_dict.items():
                epoch_loss_avgs[loss_name] = np.average(epoch_losses[loss_name])
                self.logs[loss_name].append(epoch_loss_avgs[loss_name])

            epoch_loss = epoch_loss_avgs['total_loss']

            # self.scheduler.step()
            self.epoch_num += 1
            if self.verbose:
                print('Epoch ' + str(epoch + 1), epoch_loss_avgs)

                if epoch_loss < min_loss:
                    print('Saving model')
                    self.save_model(save_dir, save_model_name, self.model)
                    self.save_auxiliary_models(save_dir)
                    min_loss = epoch_loss

                self.save_to_json(f"{save_dir}/logs.json", self.logs)

        return self.logs

    def loss(self, batch) -> dict:
        raise NotImplementedError()

    def _step_optimizer(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad)
        self.optimizer.step()

    def _log_losses(self, loss_dict):
        for loss_name, loss in loss_dict.items():
            self.logs[loss_name].append(loss.item())

    def save_auxiliary_models(self, save_dir):
        pass

    def save_model(self, save_dir, save_model_name, model):
        torch.save(model, f'{save_dir}/{save_model_name}.pt')

    def load(self, filename, directory, model):
        model.load_state_dict(torch.load('%s/%s.pt' % (directory, filename)))

    def save_to_json(self, path, obj):
        with open(path, "w") as json_file:
            json.dump(obj, json_file, indent=2)

    def _add_params_to_optimizer(self, param_list):
        params = list(self.model.parameters()) + param_list
        self.optimizer = torch.optim.Adam(params, lr=self.lr)

class TrainerBehaviorCloning(TrainerBase):

    def __init__(self, model, lr=1e-4, verbose=1):
        super().__init__(model, lr, verbose)
        self.aug_func = PendulumTranslateUniform()



    def loss(self, batch):
        state = batch['state'].type(torch.float).to(device)
        action = batch['action'].type(torch.float).to(device)
        # state, action = self.aug_func.augment(10, state, action, action, action, action, action)

        action_pred = self.model(state)
        bc_loss = F.mse_loss(action_pred, action)

        loss_dict =  {}
        loss_dict['total_loss'] = bc_loss

        return loss_dict

def train(
        env_id,
        hidden_size=256,
        n_epochs=100,
        n_samples=int(20e3),
        batch_size=64,
        lr=1e-3,
        save_root_dir=None,
        save_model_name='',
        slice=0
):

    trainer_kwargs = locals()

    print(os.getcwd())
    data = load_dataset(f'./rl/demonstrations/{env_id}/init_pos_0.npz', n=n_samples, slice=slice)

    save_root_dir = './models' if save_root_dir is None else save_root_dir
    save_dir = f'./{save_root_dir}/{env_id}'
    mkdirs(save_dir)
    print(f'save_dir = {save_dir}')

    with open(f'{save_dir}/trainer_kwargs.json', "w") as json_file:
        json.dump(trainer_kwargs, json_file, indent=2)

    robot_state_dim = data.robot_state_dim
    state_dim = data.state_dim
    action_dim = data.action_dim

    env = gym.make(env_id)

    model = TD3.load('rl/experiments/InvertedPendulum-v2/td3/init_pos_0/run_2/best_model.zip', env=env).policy
    trainer = TrainerBehaviorCloning(model=model, lr=lr, verbose=1)
    logs = trainer.train(dataset=data, save_dir=save_dir, save_model_name=save_model_name, batch_size=batch_size,epochs=n_epochs)
    plot_loss(logs, save_dir, 'loss')

    print(save_dir)

def augment(data):
    pass

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--save-root-dir", type=str, default='tmp')
    parser.add_argument("--save-model-name", type=str, default='policy')
    parser.add_argument("--env-id", type=str, default="InvertedPendulum-v2", help="environment ID")
    parser.add_argument("--n-samples", type=int, default=int(5e3))
    parser.add_argument("--n-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--slice", type=float, default=0)


    kwargs = vars(parser.parse_args())
    print(kwargs)
    train(**kwargs)

