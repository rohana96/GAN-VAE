import os

from torch import optim

from utils import *
from collections import OrderedDict
import torch.nn as nn
import matplotlib.pyplot as plt
from model import AEModel


def sample_latent(mu, log_sigma):
    sigma = torch.exp(log_sigma)
    z = sigma * torch.randn(size=mu.shape, device=sigma.device) + mu
    return z


def ae_loss(model, x):
    """ 
    TODO 2.1.2: fill in MSE loss between x and its reconstruction. 
    return loss, {recon_loss = loss} 
    """
    z = model.encoder(x)
    recon = model.decoder(z)
    criterion = nn.MSELoss(reduction='sum')
    loss = criterion(x, recon) / x.shape[0]
    return loss, OrderedDict(recon_loss=loss)


def vae_loss(model, x, beta=1):
    """TODO 2.2.2 : Fill in recon_loss and kl_loss. """

    # TODO: add reparameterization trick
    mu, log_sigma = model.encoder(x)
    z = sample_latent(mu, log_sigma)
    sigma = torch.exp(log_sigma)
    reconstruction = model.decoder(z)
    criterion = nn.MSELoss(reduction='sum')
    recon_loss = criterion(x, reconstruction) / x.shape[0]
    kl_loss = 0.5 * torch.sum(mu ** 2 + sigma ** 2 - 2 * torch.log(sigma) - 1) / x.shape[0]

    total_loss = recon_loss + beta * kl_loss
    return total_loss, OrderedDict(recon_loss=recon_loss, kl_loss=kl_loss)


def constant_beta_scheduler(target_val=1):
    def _helper(epoch):
        return target_val

    return _helper


def linear_beta_scheduler(max_epochs=None, target_val=1):
    """TODO 2.3.2 : Fill in helper. The value returned should increase linearly 
    from 0 at epoch 0 to target_val at epoch max_epochs """

    def _helper(epoch):
        return target_val * epoch / max_epochs

    return _helper


def run_train_epoch(model, loss_mode, train_loader, optimizer, beta=1, grad_clip=1):
    model.train()
    all_metrics = []
    for x, _ in train_loader:
        x = preprocess_data(x)
        if loss_mode == 'ae':
            loss, _metric = ae_loss(model, x)
        elif loss_mode == 'vae':
            loss, _metric = vae_loss(model, x, beta)
        all_metrics.append(_metric)
        optimizer.zero_grad()
        loss.backward()

        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

    return avg_dict(all_metrics)


def get_val_metrics(model, loss_mode, val_loader):
    model.eval()
    all_metrics = []
    with torch.no_grad():
        for x, _ in val_loader:
            x = preprocess_data(x)
            if loss_mode == 'ae':
                _, _metric = ae_loss(model, x)
            elif loss_mode == 'vae':
                _, _metric = vae_loss(model, x)
            all_metrics.append(_metric)

    return avg_dict(all_metrics)


def main(log_dir, loss_mode='vae', beta_mode='constant', num_epochs=20, batch_size=256, latent_size=256,
         target_beta_val=1, grad_clip=1, lr=1e-3, eval_interval=5):
    os.makedirs('data/' + log_dir, exist_ok=True)
    train_loader, val_loader = get_dataloaders()

    variational = True if loss_mode == 'vae' else False
    model = AEModel(variational, latent_size, input_shape=(3, 32, 32)).cuda()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    vis_x = next(iter(val_loader))[0][:36]

    # beta_mode is for part 2.3, you can ignore it for parts 2.1, 2.2
    if beta_mode == 'constant':
        beta_fn = constant_beta_scheduler(target_val=target_beta_val)
    elif beta_mode == 'linear':
        beta_fn = linear_beta_scheduler(max_epochs=num_epochs, target_val=target_beta_val)

    metric_list = {'recon_loss': [], 'kl_loss': []}

    for epoch in range(num_epochs):
        print('epoch', epoch)
        train_metrics = run_train_epoch(model, loss_mode, train_loader, optimizer, beta_fn(epoch))
        val_metrics = get_val_metrics(model, loss_mode, val_loader)

        # TODO : add plotting code for metrics (required for multiple parts)
        for metric, val in val_metrics.items():
            metric_list[metric].append(val)

        if (epoch + 1) % eval_interval == 0:
            print(epoch, train_metrics)
            print(epoch, val_metrics)

            vis_recons(model, vis_x, 'data/' + log_dir + '/epoch_' + str(epoch))
            if loss_mode == 'vae':
                vis_samples(model, 'data/' + log_dir + '/epoch_' + str(epoch))
    return metric_list

    # fig1, ax1 = plt.subplots()
    # ax1.set_title(f"{log_dir}")
    # ax1.set_xlabel("epoch")
    # ax1.set_ylabel("loss")
    # for metric in metric_list:
    #     if len(metric_list[metric]) > 0:            
    #         ax1.plot(range(num_epochs), metric_list[metric],  label=f'{metric}')
    # ax1.legend()
    # plt.savefig('data/' + log_dir + '/plot.png')


def plot(metric_lists, log_dir, labels):
    for metric in ['recon_loss', 'kl_loss']:
        fig1, ax1 = plt.subplots()
        ax1.set_title(f"{metric}")
        ax1.set_xlabel("epoch")
        ax1.set_ylabel("loss")
        for i, metric_list in enumerate(metric_lists):
            if len(metric_list[metric]) > 0:
                N = len(metric_list[metric])
                ax1.plot(range(N), metric_list[metric], label=labels[i])
        ax1.legend()
        plt.savefig(f'data/{log_dir}/plot_{metric}.png')
        ax1.cla()


if __name__ == '__main__':
    # TODO: Experiments to run :
    # 2.1 - Auto-Encoder
    # Run for latent_sizes 16, 128 and 1024
    n_epoch = 20
    metric_list_ae1024 = main('ae_latent1024', loss_mode='ae', num_epochs=n_epoch, latent_size=1024)
    metric_list_ae128 = main('ae_latent128', loss_mode='ae', num_epochs=n_epoch, latent_size=128)
    metric_list_ae16 = main('ae_latent16', loss_mode='ae', num_epochs=n_epoch, latent_size=16)

    labels = ['latent1024', 'latent128', 'latent16']
    plot([metric_list_ae1024, metric_list_ae128, metric_list_ae16], log_dir="plots_ae", labels=labels)

    # Q 2.2 - Variational Auto-Encoder
    metric_list_b1 = main('vae_latent1024', loss_mode='vae', num_epochs=n_epoch, latent_size=1024)

    labels = ['vanilla_vae_1024']
    plot([metric_list_b1], log_dir="plots_vanilla_vae", labels=labels)

    # Q 2.3.1 - Beta-VAE (constant beta)
    # Run for beta values 0.8, 1.2
    metric_list_b0p8 = main('vae_latent1024_beta_constant0.8', loss_mode='vae', beta_mode='constant', target_beta_val=0.8, num_epochs=n_epoch,
                            latent_size=1024)
    metric_list_b1p2 = main('vae_latent1024_beta_constant1.2', loss_mode='vae', beta_mode='constant', target_beta_val=1.2, num_epochs=n_epoch,
                            latent_size=1024)

    labels = ['beta_0.8', 'beta_1.0', 'beta_1.2']
    plot([metric_list_b0p8, metric_list_b1, metric_list_b1p2], log_dir="plots_beta_vae", labels=labels)

    # Q 2.3.2 - VAE with annealed beta (linear schedule)
    metric_list_sched = main('vae_latent1024_beta_linear1', loss_mode='vae', beta_mode='linear', target_beta_val=1, num_epochs=n_epoch,
                             latent_size=1024)

    labels = ['vae_linear_sched']
    plot([metric_list_sched], log_dir="plots_anneal_vae", labels=labels)
