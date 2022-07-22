import json
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

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

    def train(self, dataset, save_dir, save_model_name='autoencoder', batch_size=64, epochs=100):

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
                print('Epoch ' + str(epoch+1), epoch_loss_avgs)

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
        torch.save(model.state_dict(), f'{save_dir}/{save_model_name}.pt')

    def load(self, filename, directory, model):
        model.load_state_dict(torch.load('%s/%s.pt' % (directory, filename)))

    def save_to_json(self, path, obj):
        with open(path, "w") as json_file:
            json.dump(obj, json_file, indent=2)

    def _add_params_to_optimizer(self, param_list):
        params = list(self.model.parameters()) + param_list
        self.optimizer = torch.optim.Adam(params, lr=self.lr)


class TrainerQ(TrainerBase):
    def __init__(self, model, lr=1e-4, gamma=0.99):
        super().__init__(model, lr=lr)
        self.gamma = gamma
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        # self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.999)


    def loss(self, batch):
        state = batch['state'].type(torch.float).to(device)
        action = batch['action'].type(torch.float).to(device)
        reward = batch['reward'].type(torch.float).to(device)
        next_state = batch['next_state'].type(torch.float).to(device)
        next_action = batch['next_action'].type(torch.float).to(device)
        done = batch['done'].type(torch.float).to(device)

        mask = ~done.type(torch.bool)
        state = state[mask]
        action = action[mask]
        reward = reward[mask]
        next_state = next_state[mask]
        next_action = next_action[mask]

        reward = reward.view(-1, 1)
        done = done.view(-1, 1)

        q = self.model(state, action)

        # q_next = torch.zeros(size=(len(state),1), device=device)
        # q_next[mask] = self.model(next_state, next_action)
        # q_target = reward + self.gamma*q_next

        q_next = self.model(next_state, next_action)
        q_target = reward + self.gamma*q_next
        # q_target = reward + (1 - done)*self.gamma*q_next
        q_rec_loss = F.mse_loss(q, q_target)

        total_loss = q_rec_loss

        loss_dict = {
            'total_loss': total_loss,
            # 'q_rec_loss': q_rec_loss,
        }

        return loss_dict