import argparse
import json
import os

from auxiliary_models import QModel
from trainers import TrainerQ
from utils import load_dataset, plot_loss


def get_model_name(state_conditioned, beta_kl, beta_dynamics, beta_q):
    name = ""
    if state_conditioned: name += "C"
    if beta_q != 0: name += "Q"
    if beta_dynamics != 0: name += "D"
    if beta_kl != 0: name += "V"
    name += "AE"
    return name

def train(
        env_id,
        hidden_size=256,
        n_epochs=100,
        n_samples=int(20e3),
        batch_size=64,
        lr=1e-3,
        random=False,
        save_root_dir=None,
        save_model_name='best_model'
):
    trainer_kwargs = locals()
    print(os.getcwd())

    if random:
        data = load_dataset(f'./data/{env_id}/random.npz', n=n_samples)
    else:
        data = load_dataset(f'./data/{env_id}/trained.npz', n=n_samples)

    save_root_dir = './models/q' if save_root_dir is None else save_root_dir
    save_dir = f'./{save_root_dir}/{env_id}/'
    if save_dir is not None and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print(f'save_dir = {save_dir}')

    with open(f'{save_dir}/trainer_kwargs.json', "w") as json_file:
        json.dump(trainer_kwargs, json_file, indent=2)

    state_dim = data.state_dim
    action_dim = data.action_dim

    model = QModel(state_dim=state_dim, action_dim=action_dim, hidden_size=hidden_size)
    trainer = TrainerQ(model=model, lr=lr)
    logs = trainer.train(dataset=data, save_dir=save_dir, save_model_name=save_model_name, batch_size=batch_size, epochs=n_epochs)
    plot_loss(logs, save_dir, 'loss')

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--save-root-dir", type=str, default='.')
    parser.add_argument("--save-model-name", type=str, default='q_true')
    parser.add_argument("--env-id", type=str, default="CartPole-v1")
    parser.add_argument("--n-samples", type=int, default=int(10e3))
    parser.add_argument("--n-epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--random", type=bool, default=False)

    kwargs = vars(parser.parse_args())
    print(kwargs)
    train(**kwargs)