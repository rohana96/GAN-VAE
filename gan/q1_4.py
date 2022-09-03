import os

import torch

from networks import Discriminator, Generator
from train import train_model


def compute_discriminator_loss(
        disc, discrim_real, discrim_fake, lamb
):
    # TODO 1.4.1: Implement LSGAN loss for discriminator.
    # Do not use discrim_interp, interp, lamb. They are placeholders for Q1.5.
    criterion = torch.nn.MSELoss()
    discrim_real = discrim_real.reshape(-1)
    loss_disc_real = criterion(discrim_real, torch.ones_like(discrim_real))
    disc_fake = discrim_fake.reshape(-1)
    loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
    loss_disc = (loss_disc_real + loss_disc_fake) / 2
    return loss_disc


def compute_generator_loss(discrim_fake):
    # TODO 1.4.1: Implement LSGAN loss for generator.
    criterion = torch.nn.MSELoss()
    loss_gen = criterion(discrim_fake, torch.ones_like(discrim_fake))
    return loss_gen


if __name__ == "__main__":
    gen = Generator().cuda().to(memory_format=torch.channels_last)
    disc = Discriminator().cuda().to(memory_format=torch.channels_last)
    prefix = "data_ls_gan_2/"
    os.makedirs(prefix, exist_ok=True)
    print(" Training lsgan ... ")
    # TODO 1.4.2: Run this line of code.
    train_model(
        gen,
        disc,
        num_iterations=int(3e4),
        batch_size=256,
        prefix=prefix,
        gen_loss_fn=compute_generator_loss,
        disc_loss_fn=compute_discriminator_loss,
        log_period=1000,
        wgan_gp=False
    )
