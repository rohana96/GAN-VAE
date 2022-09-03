import os
import torch

from networks import Discriminator, Generator
from train import train_model
from utils import get_device


def gradient_penalty(real, fake, disc, epsilon):

    interp = real * epsilon + fake * (1 - epsilon)
    disc_interp = disc(interp)
    gradient = torch.autograd.grad(
        inputs=interp,
        outputs=disc_interp,
        grad_outputs=torch.ones_like(disc_interp),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gp = torch.mean((gradient_norm - 1) ** 2)
    return gp

def compute_discriminator_loss(
        real, fake, disc, discrim_real, discrim_fake, lamb
):
    # TODO 1.5.1: Implement WGAN-GP loss for discriminator.
    # loss = max_D E[D(real_data)] - E[D(fake_data)] + lambda * E[(|| grad wrt interpolated_data (D(interpolated_data))|| - 1)^2]
    B, C, H, W = real.shape
    device = get_device()
    epsilon = torch.rand(size=(B, 1, 1, 1)).repeat(1, C, H, W).to(device)
    gp = gradient_penalty(real, fake, disc, epsilon)
    loss_discrim = -(torch.mean(discrim_real) - torch.mean(discrim_fake)) + lamb * gp
    return loss_discrim


def compute_generator_loss(discrim_fake):
    # TODO 1.5.1: Implement WGAN-GP loss for generator.
    # loss = - E[D(fake_data)]
    # output = disc(discrim_fake)
    loss_gen = -torch.mean(discrim_fake)
    return loss_gen


if __name__ == "__main__":
    gen = Generator().cuda().to(memory_format=torch.channels_last)
    disc = Discriminator().cuda().to(memory_format=torch.channels_last)
    prefix = "data_wgan_gp/"
    os.makedirs(prefix, exist_ok=True)
    print(" Training wgap-gp ... ")
    # TODO 1.5.2: Run this line of code.
    train_model(
        gen,
        disc,
        num_iterations=int(3e4),
        batch_size=256,
        prefix=prefix,
        gen_loss_fn=compute_generator_loss,
        disc_loss_fn=compute_discriminator_loss,
        log_period=1000,
        wgan_gp=True
    )
