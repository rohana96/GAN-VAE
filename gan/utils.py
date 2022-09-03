import torch
from cleanfid import fid
from matplotlib import pyplot as plt
import numpy as np
import torchvision
import torchvision.transforms.functional as F


def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


def save_plot(x, y, xlabel, ylabel, title, filename):
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(filename + ".png")


@torch.no_grad()
def get_fid(gen, dataset_name, dataset_resolution, z_dimension, batch_size, num_gen):
    gen_fn = lambda z: (gen.forward_given_samples(z) / 2 + 0.5) * 255
    score = fid.compute_fid(
        gen=gen_fn,
        dataset_name=dataset_name,
        dataset_res=dataset_resolution,
        num_gen=num_gen,
        z_dim=z_dimension,
        batch_size=batch_size,
        verbose=True,
        dataset_split="custom",
    )
    return score

def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    

@torch.no_grad()
def interpolate_latent_space(gen, fixed_noise, path):
    # TODO 1.2: Generate and save out latent space interpolations.
    # Concretely, for the first two dimensions of the latent space
    # generate a grid of points that range from -1 to 1 on each dimension (10 points for each dimension).
    # hold the rest of z to be some fixed random value. Forward the generated samples through the generator
    # and save out an image holding all 100 samples.
    # use torchvision.utils.save_image to save out the visualization.

    # http://pytorch.org/vision/main/auto_examples/plot_visualization_utils.html#sphx-glr-auto-examples-plot-visualization-utils-py

    images = torch.clamp(gen.forward_given_samples(fixed_noise) * 0.5 + 0.5, min=0, max=1)
    print(images.shape)
    grid = torchvision.utils.make_grid(images, nrow=10)
    torchvision.utils.save_image(grid, path, normalize=False)


def test_interpolate_latent_space():

    z1s = torch.linspace(-1, 1, steps=10)
    z2s = torch.linspace(-1, 1, steps=10)
    fixed_noise = torch.randn(32, 3*100*100)
    fixed_noise_ = fixed_noise[0].repeat(4, 1) # generate 2 x 2 grid 

    for i, z1 in enumerate(z1s):
        for j, z2 in enumerate(z2s):
            fixed_noise_[2*i + j, 0] = z1
            fixed_noise_[2*i + j, 1] = z2

    images = fixed_noise_.reshape((-1,3, 100, 100))
    print(images.shape)
    grid = torchvision.utils.make_grid(images, nrow=10)
    torchvision.utils.save_image(grid, path, normalize=False)


if __name__ == "__main__":
    test_interpolate_latent_space()