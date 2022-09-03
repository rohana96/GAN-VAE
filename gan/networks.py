import torch
import torch.jit as jit
import torch.nn as nn


class UpSampleConv2D(jit.ScriptModule):
    # TODO 1.1: Implement nearest neighbor upsampling + conv layer

    def __init__(
            self,
            input_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            n_filters=128,
            upscale_factor=2,
            padding=0,
    ):
        super(UpSampleConv2D, self).__init__()
        self.upscale_factor = upscale_factor
        self.conv = nn.Conv2d(
            in_channels=input_channels,
            out_channels=n_filters,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        self.pixel_shuffle = nn.PixelShuffle(self.upscale_factor)

    @jit.script_method
    def forward(self, x):
        # TODO 1.1: Implement nearest neighbor upsampling
        r = self.upscale_factor
        out = x.repeat(1, r * r, 1, 1)
        out = self.pixel_shuffle(out)
        out = self.conv(out)
        return out
        # 1. Stack x channel wise upscale_factor^2 times
        # 2. Then re-arrange to form a batch x channel x height*upscale_factor x width*upscale_factor
        # 3. Apply convolution.
        # Hint for 2. look at
        # https://pytorch.org/docs/master/generated/torch.nn.PixelShuffle.html#torch.nn.PixelShuffle


def test_upsample_conv():
    x = torch.randn(size=(10, 3, 100, 100))
    upsampleconv = UpSampleConv2D(3)
    print(upsampleconv(x).shape)


class DownSampleConv2D(jit.ScriptModule):
    # TODO 1.1: Implement spatial mean pooling + conv layer

    def __init__(
            self, input_channels, kernel_size=(3, 3), stride=(1, 1), n_filters=128, downscale_ratio=2, padding=0
    ):
        super(DownSampleConv2D, self).__init__()
        self.downscale_ratio = downscale_ratio
        self.conv = nn.Conv2d(
            in_channels=input_channels,
            out_channels=n_filters,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        self.pixel_unshuffle = nn.PixelUnshuffle(self.downscale_ratio)

    @jit.script_method
    def forward(self, x):
        # TODO 1.1: Implement spatial mean pooling
        d = self.downscale_ratio
        out = self.pixel_unshuffle(x)
        B, C, H, W = out.shape
        out = out.view(B, d * d, -1, H, W)
        out = out.mean(dim=1)
        out = self.conv(out)
        return out
        # 1. Re-arrange to form a batch x channel * upscale_factor^2 x height x width
        # 2. Then split channel wise into batch x channel x height x width Images
        # 3. average the images into one and apply convolution
        # Hint for 1. look at
        # https://pytorch.org/docs/master/generated/torch.nn.PixelUnshuffle.html#torch.nn.PixelUnshuffle


def test_downsample_conv():
    x = torch.randn(size=(2, 1, 16, 16))
    downsampleconv = DownSampleConv2D(1)
    print(downsampleconv(x).shape)


class ResBlockUp(jit.ScriptModule):
    # TODO 1.1: Implement Residual Block Upsampler.
    """
    ResBlockUp(
        (layers): Sequential(
            (0): BatchNorm2d(in_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (1): ReLU()
            (2): Conv2d(in_channels, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (3): BatchNorm2d(n_filters, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (4): ReLU()
        )
        (residual): UpSampleConv2D(
            (conv): Conv2d(n_filters, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (shortcut): UpSampleConv2D(
            (conv): Conv2d(n_filters, n_filters, kernel_size=(1, 1), stride=(1, 1))
        )
    """

    def __init__(self, input_channels, kernel_size=(3, 3), n_filters=128):
        super(ResBlockUp, self).__init__()
        self.layers = nn.Sequential(
            nn.BatchNorm2d(
                num_features=input_channels,
                eps=1e-05,
                momentum=0.1,
                affine=True,
                track_running_stats=True
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=n_filters,
                kernel_size=kernel_size,
                stride=(1, 1),
                padding=(1, 1),
                bias=False
            ),
            nn.BatchNorm2d(
                num_features=n_filters,
                eps=1e-05,
                momentum=0.1,
                affine=True,
                track_running_stats=True
            ),
            nn.ReLU(),
        )

        self.residual = UpSampleConv2D(
            input_channels=n_filters,
            kernel_size=kernel_size,
            n_filters=n_filters,
            padding=1,
            upscale_factor=2,
        )

        self.shortcut = UpSampleConv2D(
            input_channels=input_channels,
            kernel_size=(1, 1),
            n_filters=n_filters,
            upscale_factor=2,
            padding=0
        )

    @jit.script_method
    def forward(self, x):
        # TODO 1.1: Forward through the layers and implement a residual connection.
        # Apply self.residual to the output of self.layers and apply self.shortcut to the original input.
        out = self.layers(x)
        out = self.residual(out)
        out = out + self.shortcut(x)
        return out


def test_resblock_up():
    x = torch.randn(size=(1, 3, 5, 5))
    resblockup = ResBlockUp(3)
    print(resblockup(x).shape)


class ResBlockDown(jit.ScriptModule):
    # TODO 1.1: Implement Residual Block Downsampler.

    def __init__(self, input_channels, kernel_size=(3, 3), n_filters=128):
        super(ResBlockDown, self).__init__()
        self.layers = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=n_filters,
                kernel_size=kernel_size,
                stride=(1, 1),
                padding=(1, 1),
                bias=False
            ),
            nn.ReLU(),
        )

        self.residual = DownSampleConv2D(
            input_channels=n_filters,
            kernel_size=kernel_size,
            n_filters=n_filters,
            padding=1,
            downscale_ratio=2,
        )

        self.shortcut = DownSampleConv2D(
            input_channels=input_channels,
            kernel_size=(1, 1),
            n_filters=n_filters,
            downscale_ratio=2,
            padding=0
        )

    @jit.script_method
    def forward(self, x):
        # TODO 1.1: Forward through the layers and implement a residual connection.
        # Apply self.residual to the output of self.layers and apply self.shortcut to the original input.
        out = self.layers(x)
        out = self.residual(out)
        out = out + self.shortcut(x)
        return out


def test_resblock_down():
    x = torch.randn(size=(1, 3, 16, 16))
    resblockdown = ResBlockDown(3)
    print(resblockdown(x).shape)


class ResBlock(jit.ScriptModule):
    # TODO 1.1: Impement Residual Block as described below.
    """
    ResBlock(
        (layers): Sequential(
            (0): ReLU()
            (1): Conv2d(in_channels, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (2): ReLU()
            (3): Conv2d(n_filters, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
    )
    """

    def __init__(self, input_channels, kernel_size=(3, 3), n_filters=128):
        super(ResBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=n_filters,
                kernel_size=kernel_size,
                stride=(1, 1),
                padding=(1, 1),
                bias=True
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=n_filters,
                out_channels=n_filters,
                kernel_size=kernel_size,
                stride=(1, 1),
                padding=(1, 1),
                bias=True
            )
        )

    @jit.script_method
    def forward(self, x):
        # TODO 1.1: Forward the conv layers. Don't forget the residual connection!
        out = self.layers(x)
        out = out + x
        return out


def test_resblock():
    x = torch.randn(size=(1, 3, 16, 16))
    resblock = ResBlock(3)
    print(resblock(x).shape)


class Generator(jit.ScriptModule):
    # TODO 1.1: Implement Generator. Follow the architecture described below:
    """
    Generator(
    (dense): Linear(in_features=128, out_features=2048, bias=True)
    (layers): Sequential(
        (0): ResBlockUp(
        (layers): Sequential(
            (0): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (1): ReLU()
            (2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (4): ReLU()
        )
        (residual): UpSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (shortcut): UpSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
        )
        )
        (1): ResBlockUp(
            (layers): Sequential(
                (0): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (1): ReLU()
                (2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (4): ReLU()
        )
        (residual): UpSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (shortcut): UpSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
        )
        )
        (2): ResBlockUp(
            (layers): Sequential(
                (0): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (1): ReLU()
                (2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (4): ReLU()
        )
        (residual): UpSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (shortcut): UpSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
        )
        )
        (3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (4): ReLU()
        (5): Conv2d(128, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (6): Tanh()
    )
    )
    """

    def __init__(self, starting_image_size=4):
        super(Generator, self).__init__()
        self.size = starting_image_size
        self.dense = nn.Linear(in_features=128, out_features=2048, bias=True)
        self.layers = nn.Sequential(
            ResBlockUp(input_channels=128),
            ResBlockUp(input_channels=128),
            ResBlockUp(input_channels=128),
            nn.BatchNorm2d(num_features=128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Tanh()
        )

    @jit.script_method
    def forward_given_samples(self, z):
        # TODO 1.1: forward the generator assuming a set of samples z have been passed in.
        # Don't forget to re-shape the output of the dense layer into an image with the appropriate size!
        out = self.dense(z)
        out = out.view(-1, 128, self.size, self.size)
        out = self.layers(out)
        return out

    @jit.script_method
    def forward(self, n_samples: int = 1024):
        # TODO 1.1: Generate n_samples latents and forward through the network.
        # Make sure to cast the latents to type half (for compatibility with torch.cuda.amp.autocast)
        z = torch.randn(size=(n_samples, 128)).half().cuda()
        out = self.forward_given_samples(z)
        return out


def test_generator():
    gen = Generator()
    print(gen(1).shape)


class Discriminator(jit.ScriptModule):
    # TODO 1.1: Impement Discriminator. Follow the architecture described below:
    """
    Discriminator(
    (layers): Sequential(
        (0): ResBlockDown(
            (layers): Sequential(
                (0): ReLU()
                (1): Conv2d(3, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (2): ReLU()
            )
            (residual): DownSampleConv2D(
                (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
            (shortcut): DownSampleConv2D(
                (conv): Conv2d(3, 128, kernel_size=(1, 1), stride=(1, 1))
            )
        )
        (1): ResBlockDown(
            (layers): Sequential(
                (0): ReLU()
                (1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (2): ReLU()
            )
            (residual): DownSampleConv2D(
                (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
            (shortcut): DownSampleConv2D(
                (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
            )
        )
        (2): ResBlock(
            (layers): Sequential(
                (0): ReLU()
                (1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (2): ReLU()
                (3): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        )
        (3): ResBlock(
            (layers): Sequential(
                (0): ReLU()
                (1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (2): ReLU()
                (3): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        )
        (4): ReLU()
    )
    (dense): Linear(in_features=128, out_features=1, bias=True)
    )
    """

    def __init__(self):
        super(Discriminator, self).__init__()
        self.layers = nn.Sequential(
            ResBlockDown(input_channels=3),
            ResBlockDown(input_channels=128),
            ResBlock(input_channels=128),
            ResBlock(input_channels=128),
            nn.ReLU()
        )
        self.dense = nn.Linear(in_features=128, out_features=1, bias=True)

    @jit.script_method
    def forward(self, x):
        # TODO 1.1: Forward the discriminator assuming a batch of images have been passed in.
        # Make sure to flatten the output of the convolutional layers before passing to the output layer!
        out = self.layers(x)
        B, C, H, W = out.shape  # B x 128 x 8 x 8
        out = out.view(B, C, -1)
        out = out.sum(dim=2)
        out = self.dense(out)
        return out


def test_discriminator():
    x = torch.randn(size=(3, 3, 32, 32))
    disc = Discriminator()
    print(disc(x).shape)


def test_channelwise_split():
    """
    (B x (nxC)) input (Every channel is 0 dimensional here)
    We say (1, 2) represent a single unit composed of two channels
    We want to split every batch entry into 5 units and then add all units channel-wise.
    All 1's get added , all 2's get added, and same for the second batch entry.

     [[1, 2, 1, 2, 1, 2, 1, 2], [3, 4, 3, 4, 3, 4, 3, 4]]  -->
     [[[1, 2], [1, 2], [1, 2], [1, 2]], [[3, 4], [3, 4], [3, 4], [3, 4]]]  -->
     [[4, 8], [12, 16]]
    """
    a = torch.Tensor([[1, 2, 1, 2, 1, 2, 1, 2], [3, 4, 3, 4, 3, 4, 3, 4]])
    a = a.view(2, 4, 2)
    a = a.sum(dim=1)
    print(a)


if __name__ == "__main__":

    # test_upsample_conv()
    # test_downsample_conv()
    # test_channelwise_split()
    # test_resblock_up()
    # test_resblock_down()
    # test_resblock()
    # test_generator()
    test_discriminator()
