import numpy as np
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_shape, latent_dim):
        super().__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        """
        TODO 2.1.1 : Fill in self.convs following the given architecture 
         Sequential(
                (0): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (1): ReLU()
                (2): Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
                (3): ReLU()
                (4): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
                (5): ReLU()
                (6): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
            )
        """
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
        )
        self.conv_out_dim = input_shape[1] // 8 * input_shape[2] // 8 * 256
        # TODO 2.1.1: fill in self.fc, such that output dimension is self.latent_dim
        self.fc = nn.Linear(self.conv_out_dim, self.latent_dim, bias=True)

    def forward(self, x):
        x = x.to('cuda')
        out = self.convs(x)
        out = out.view(out.shape[0], -1)
        out = self.fc(out)
        return out
        # TODO 2.1.1 : forward pass through the network, output should be of dimension : self.latent_dim


def test_encoder():
    x = torch.randn(size=(1, 3, 32, 32))
    input_shape = x.shape[-3:]
    model = Encoder(input_shape, 10)
    print(model(x).shape)


class VAEEncoder(Encoder):
    def __init__(self, input_shape, latent_dim):
        super().__init__(input_shape, latent_dim)
        # TODO 2.2.1: fill in self.fc, such that output dimension is 2*self.latent_dim
        self.fc_mu = nn.Linear(latent_dim, latent_dim)
        self.fc_log_sigma = nn.Linear(latent_dim, latent_dim)

    def forward(self, x):
        # TODO 2.2.1: forward pass through the network.
        # should return a tuple of 2 tensors, each of dimension self.latent_dim
        out = self.convs(x)
        out = out.view(out.shape[0], -1)
        out = self.fc(out)
        out_mu = self.fc_mu(out)
        out_log_sigma = self.fc_log_sigma(out)
        return out_mu, out_log_sigma


def test_vae_encoder():
    x = torch.randn(size=(1, 3, 32, 32))
    input_shape = x.shape[-3:]
    model = VAEEncoder(input_shape, 10)
    mu, log_sigma = model(x)
    print(mu.shape, log_sigma.shape)


class Decoder(nn.Module):
    def __init__(self, latent_dim, output_shape):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_shape = output_shape

        # TODO 2.1.1: fill in self.base_size
        self.base_size = (256, 4, 4)
        self.fc = nn.Linear(latent_dim, np.prod(self.base_size))

        """
        TODO 2.1.1 : Fill in self.deconvs following the given architecture 
        Sequential(
                (0): ReLU()
                (1): ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
                (2): ReLU()
                (3): ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
                (4): ReLU()
                (5): ConvTranspose2d(64, 32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
                (6): ReLU()
                (7): Conv2d(32, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        """
        self.deconvs = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )

    def forward(self, z):
        # TODO 2.1.1: forward pass through the network, first through self.fc, then self.deconvs.
        out = self.fc(z)
        C, H, W = self.base_size
        out = out.view(-1, C, H, W)
        out = self.deconvs(out)
        return out
    

def test_decoder():
    z = torch.randn((32, 128))
    decoder = Decoder(128, 32)
    out = decoder(z)
    print(out.shape)



class AEModel(nn.Module):
    def __init__(self, variational, latent_size, input_shape=(3, 32, 32)):
        super().__init__()
        assert len(input_shape) == 3

        self.input_shape = input_shape
        self.latent_size = latent_size
        if variational:
            self.encoder = VAEEncoder(input_shape, latent_size)
        else:
            self.encoder = Encoder(input_shape, latent_size)
        self.decoder = Decoder(latent_size, input_shape)

if __name__ == "__main__":
    # test_encoder()
    # test_vae_encoder()
    test_decoder()