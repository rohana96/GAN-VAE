import os
from glob import glob
import tqdm
import torch
import math
from PIL import Image
from torchvision import transforms as T
from torchvision.datasets import VisionDataset
from torchvision.utils import save_image
from utils import get_fid, interpolate_latent_space, save_plot, get_device
import random
import time
# random.seed(10)

def build_transforms():
    # TODO 1.2: Add two transforms:
    # 1. Convert input image to tensor.
    # 2. Rescale input image to be between -1 and 1.
    ds_transforms = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    return ds_transforms


def get_optimizers_and_schedulers(gen, disc):
    # TODO 1.2 Get optimizers and learning rate schedulers.
    # 1. Construct the optimizers for the discriminator and generator.
    # Both should use the Adam optimizer with learning rate of .0002 and Beta1 = 0, Beta2 = 0.9.
    # 2. Construct the learning rate schedulers for the generator and discriminator.
    # The learning rate for the discriminator should be decayed to 0 over 500K steps.
    # The learning rate for the generator should be decayed to 0 over 100K steps.

    optim_discriminator = torch.optim.Adam(disc.parameters(), lr=0.0002, betas=(0.0, 0.9))
    # scheduler_discriminator = torch.optim.lr_scheduler.StepLR(optim_discriminator, step_size=300, gamma=0.1)
    scheduler_discriminator = torch.optim.lr_scheduler.LinearLR(
        optim_discriminator, 
        start_factor=1.00, 
        end_factor=0.0, 
        total_iters=500000, 
        last_epoch=-1, 
        verbose=False
        )
        
    optim_generator = torch.optim.Adam(gen.parameters(), lr=0.0002, betas=(0.0, 0.9))
    # scheduler_generator = torch.optim.lr_scheduler.StepLR(optim_generator, step_size=300, gamma=0.1)
    scheduler_generator = torch.optim.lr_scheduler.LinearLR(
        optim_generator, 
        start_factor=1.00, 
        end_factor=0.0, 
        total_iters=100000, 
        last_epoch=-1, 
        verbose=False
        )

    return (
        optim_discriminator,
        scheduler_discriminator,
        optim_generator,
        scheduler_generator,
    )


class Dataset(VisionDataset):
    def __init__(self, root, transform=None):
        super(Dataset, self).__init__(root)
        self.file_names = glob(os.path.join(self.root, "*.jpg"), recursive=True)
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(self.file_names[index])
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.file_names)


def train_model(
        gen,
        disc,
        num_iterations,
        batch_size,
        lamb=10,
        prefix=None,
        gen_loss_fn=None,
        disc_loss_fn=None,
        log_period=10000,
        wgan_gp=False
):
    torch.backends.cudnn.benchmark = True
    ds_transforms = build_transforms()
    train_loader = torch.utils.data.DataLoader(
        Dataset(root="datasets/CUB_200_2011_32", transform=ds_transforms),
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    (
        optim_discriminator,
        scheduler_discriminator,
        optim_generator,
        scheduler_generator,
    ) = get_optimizers_and_schedulers(gen, disc)
    scaler = torch.cuda.amp.GradScaler()
    device = get_device()
    iters = 0
    fids_list = []
    iters_list = []
    fixed_noise = torch.randn(40, 128).to(device)

    # Compute fixed latent space codes
    z1s = torch.linspace(-1, 1, steps=10)
    z2s = torch.linspace(-1, 1, steps=10)
    fixed_noise_ = fixed_noise[0].repeat(100, 1)

    for i, z1 in enumerate(z1s):
        for j, z2 in enumerate(z2s):
            fixed_noise_[10*i + j, 0] = z1
            fixed_noise_[10*i + j, 1] = z2

    min_fid = math.inf
    while iters < num_iterations:
        for train_batch in tqdm.tqdm(train_loader):
            with torch.cuda.amp.autocast():
                train_batch = train_batch.cuda()
                # TODO 1.2: compute generator outputs and discriminator outputs
                real = train_batch
                # 1. Compute generator output -> the number of samples must match the batch size.
                # 2. Compute discriminator output on the train batch.
                # 3. Compute the discriminator output on the generated data.
                # start_time = time.time()
                fake = gen(n_samples=real.shape[0])
                discrim_real = disc(real)
                discrim_fake = disc(fake)
                # end_time = time.time()

                # print(f"took {end_time - start_time} seconds for discriminator forward pass for iter {i}")
                # TODO: 1.5 Compute the interpolated batch and run the discriminator on it.
                """Implemented in q1.5"""
                # To compute interpolated data, draw eps ~ Uniform(0, 1)
                # interpolated data = eps * fake_data + (1-eps) * real_data
                # eps = torch.randn()
                # interp = None
                # discrim_interp = None
                if not wgan_gp:
                    discriminator_loss = disc_loss_fn(disc, discrim_real, discrim_fake, lamb)
                else:
                    discriminator_loss = disc_loss_fn(real, fake, disc, discrim_real, discrim_fake, lamb)
            
            # start_time = time.time()
            optim_discriminator.zero_grad(set_to_none=True)
            scaler.scale(discriminator_loss).backward()
            scaler.step(optim_discriminator)
            scheduler_discriminator.step()
            # end_time = time.time()
            # print(f"took {end_time - start_time} seconds for discriminator backward pass for iter {i}")
            if iters % 5 == 0:
                with torch.cuda.amp.autocast():
                    # TODO 1.2: Compute samples and evaluate under discriminator.
                    # with torch.no_grad():
                    fake = gen(n_samples=batch_size)
                    discrim_fake = disc(fake)
                    generator_loss = gen_loss_fn(discrim_fake)

                optim_generator.zero_grad(set_to_none=True)
                scaler.scale(generator_loss).backward()
                scaler.step(optim_generator)
                scheduler_generator.step()

            if iters % log_period == 0 and iters != 0:
                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        # TODO 1.2: Generate samples using the generator, make sure they lie in the range [0, 1].
                        generated_samples = torch.clamp(gen.forward_given_samples(fixed_noise) * 0.5 + 0.5, min=0, max=1)

                    save_image(
                        generated_samples.data.float(),
                        prefix + "samples_{}.png".format(iters),
                        nrow=10,
                    )
                    torch.jit.save(gen, prefix + "/generator.pt")
                    torch.jit.save(disc, prefix + "/discriminator.pt")
                    fid = fid = get_fid(
                        gen,
                        dataset_name="cub",
                        dataset_resolution=32,
                        z_dimension=128,
                        batch_size=256,
                        num_gen=10_000,
                    )

                    if fid < min_fid:
                        min_fid = fid
                        torch.jit.save(gen, prefix + f"/generator_min_fid.pt")
                        torch.jit.save(disc, prefix + f"/discriminator_min_fid.pt")  

                    print(f"Iteration {iters} FID: {fid}")
                    fids_list.append(fid)
                    iters_list.append(iters)

                    save_plot(
                        iters_list,
                        fids_list,
                        xlabel="Iterations",
                        ylabel="FID",
                        title="FID vs Iterations",
                        filename=prefix + "fid_vs_iterations",
                    )
                    interpolate_latent_space(
                        gen, fixed_noise_, prefix + "interpolations_{}.png".format(iters)
                    )
            scaler.update()
            iters += 1
    fid = get_fid(
        gen,
        dataset_name="cub",
        dataset_resolution=32,
        z_dimension=128,
        batch_size=256,
        num_gen=50_000,
    )
    print(f"Final FID (Full 50K): {fid}")