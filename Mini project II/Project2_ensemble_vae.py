# Code for DTU course 02460 (Advanced Machine Learning Spring) by Jes Frellsen, 2024
# Version 1.0 (2024-01-27)
# Inspiration is taken from:
# - https://github.com/jmtomczak/intro_dgm/blob/main/vaes/vae_example.ipynb
# - https://github.com/kampta/pytorch-distributions/blob/master/gaussian_vae.py
#
# Significant extension by Søren Hauberg, 2024

# To run it
# python Project2_ensemble_vae.py geodesics --device cuda --num-curves 25 --num-t 15
# To test it
# python Project2_ensemble_vae.py geodesics --device cuda --num-curves 2 --num-t 3
# To save an image:
# save_image(torch.cat([data.cpu(), recon.cpu()], dim=0), "something.png")
# TODO: Add all the 2048 points 

############
import torch
import torch.nn as nn
import torch.distributions as td
import torch.utils.data
from tqdm import tqdm
from copy import deepcopy
import os
import math
import matplotlib.pyplot as plt
from geodesics_utils import optimize_geodesic, plot_geodesic_latents,plot_decoded_images,plot_latent_std_background
import matplotlib.lines as mlines


class GaussianPrior(nn.Module):
    def __init__(self, M):
        """
        Define a Gaussian prior distribution with zero mean and unit variance.

                Parameters:
        M: [int]
           Dimension of the latent space.
        """
        super(GaussianPrior, self).__init__()
        self.M = M
        self.mean = nn.Parameter(torch.zeros(self.M), requires_grad=False)
        self.std = nn.Parameter(torch.ones(self.M), requires_grad=False)

    def forward(self):
        """
        Return the prior distribution.

        Returns:
        prior: [torch.distributions.Distribution]
        """
        return td.Independent(td.Normal(loc=self.mean, scale=self.std), 1)


class GaussianEncoder(nn.Module):
    def __init__(self, encoder_net):
        """
        Define a Gaussian encoder distribution based on a given encoder network.

        Parameters:
        encoder_net: [torch.nn.Module]
           The encoder network that takes as a tensor of dim `(batch_size,
           feature_dim1, feature_dim2)` and output a tensor of dimension
           `(batch_size, 2M)`, where M is the dimension of the latent space.
        """
        super(GaussianEncoder, self).__init__()
        self.encoder_net = encoder_net

    def forward(self, x):
        """
        Given a batch of data, return a Gaussian distribution over the latent space.

        Parameters:
        x: [torch.Tensor]
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        mean, std = torch.chunk(self.encoder_net(x), 2, dim=-1)
        return td.Independent(td.Normal(loc=mean, scale=torch.exp(std)), 1)


class GaussianDecoder(nn.Module):
    def __init__(self, decoder_net):
        """
        Define a Bernoulli decoder distribution based on a given decoder network.

        Parameters:
        encoder_net: [torch.nn.Module]
           The decoder network that takes as a tensor of dim `(batch_size, M) as
           input, where M is the dimension of the latent space, and outputs a
           tensor of dimension (batch_size, feature_dim1, feature_dim2).
        """
        super(GaussianDecoder, self).__init__()
        self.decoder_net = decoder_net
        # self.std = nn.Parameter(torch.ones(28, 28) * 0.5, requires_grad=True) # In case you want to learn the std of the gaussian.

    def forward(self, z):
        """
        Given a batch of latent variables, return a Bernoulli distribution over the data space.

        Parameters:
        z: [torch.Tensor]
           A tensor of dimension `(batch_size, M)`, where M is the dimension of the latent space.
        """
        means = self.decoder_net(z)
        return td.Independent(td.Normal(loc=means, scale=1e-1), 3)


class VAE(nn.Module):
    """
    Define a Variational Autoencoder (VAE) model.
    """

    def __init__(self, prior, decoder, encoder):
        """
        Parameters:
        prior: [torch.nn.Module]
           The prior distribution over the latent space.
        decoder: [torch.nn.Module]
              The decoder distribution over the data space.
        encoder: [torch.nn.Module]
                The encoder distribution over the latent space.
        """

        super(VAE, self).__init__()
        self.prior = prior
        self.decoder = decoder
        self.encoder = encoder

    def elbo(self, x):
        """
        Compute the ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor]
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2, ...)`
           n_samples: [int]
           Number of samples to use for the Monte Carlo estimate of the ELBO.
        """
        q = self.encoder(x)
        z = q.rsample()

        elbo = torch.mean(
            self.decoder(z).log_prob(x) - q.log_prob(z) + self.prior().log_prob(z)
        )
        return elbo

    def sample(self, n_samples=1):
        """
        Sample from the model.

        Parameters:
        n_samples: [int]
           Number of samples to generate.
        """
        z = self.prior().sample(torch.Size([n_samples]))
        return self.decoder(z).sample()

    def forward(self, x):
        """
        Compute the negative ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor]
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        return -self.elbo(x)


def train(model, optimizer, data_loader, epochs, device):
    """
    Train a VAE model.

    Parameters:
    model: [VAE]
       The VAE model to train.
    optimizer: [torch.optim.Optimizer]
         The optimizer to use for training.
    data_loader: [torch.utils.data.DataLoader]
            The data loader to use for training.
    epochs: [int]
        Number of epochs to train for.
    device: [torch.device]
        The device to use for training.
    """

    num_steps = len(data_loader) * epochs
    epoch = 20

    def noise(x, std=0.05):
        eps = std * torch.randn_like(x)
        return torch.clamp(x + eps, min=0.0, max=1.0)

    with tqdm(range(num_steps)) as pbar:
        for step in pbar:
            try:
                x = next(iter(data_loader))[0]
                x = noise(x.to(device))
                model = model
                optimizer.zero_grad()
                # from IPython import embed; embed()
                loss = model(x)
                loss.backward()
                optimizer.step()

                # Report
                if step % 5 == 0:
                    loss = loss.detach().cpu()
                    pbar.set_description(
                        f"total epochs ={epoch}, step={step}, loss={loss:.1f}"
                    )

                if (step + 1) % len(data_loader) == 0:
                    epoch += 1
            except KeyboardInterrupt:
                print(
                    f"Stopping training at total epoch {epoch} and current loss: {loss:.1f}"
                )
                break


if __name__ == "__main__":
    from torchvision import datasets, transforms
    from torchvision.utils import save_image

    # Parse arguments
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        type=str,
        default="train",
        choices=["train", "sample", "eval", "geodesics"],
        help="what to do when running the script (default: %(default)s)",
    )
    parser.add_argument(
        "--experiment-folder",
        type=str,
        default="experiment",
        help="folder to save and load experiment results in (default: %(default)s)",
    )
    parser.add_argument(
        "--samples",
        type=str,
        default="samples.png",
        help="file to save samples in (default: %(default)s)",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="torch device (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        metavar="N",
        help="batch size for training (default: %(default)s)",
    )
    parser.add_argument(
        "--epochs-per-decoder",
        type=int,
        default=50,
        metavar="N",
        help="number of training epochs per each decoder (default: %(default)s)",
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=2,
        metavar="N",
        help="dimension of latent variable (default: %(default)s)",
    )
    parser.add_argument(
        "--num-decoders",
        type=int,
        default=1,
        metavar="N",
        help="number of decoders in the ensemble (default: %(default)s)",
    )
    parser.add_argument(
        "--num-reruns",
        type=int,
        default=10,
        metavar="N",
        help="number of reruns (default: %(default)s)",
    )
    parser.add_argument(
        "--num-curves",
        type=int,
        default=10,
        metavar="N",
        help="number of geodesics to plot (default: %(default)s)",
    )
    parser.add_argument(
        "--num-t",  # number of points along the curve
        type=int,
        default=20,
        metavar="N",
        help="number of points along the curve (default: %(default)s)",
    )

    args = parser.parse_args()
    print("# Options")
    for key, value in sorted(vars(args).items()):
        print(key, "=", value)

    device = args.device

    # Load a subset of MNIST and create data loaders
    def subsample(data, targets, num_data, num_classes):
        idx = targets < num_classes
        new_data = data[idx][:num_data].unsqueeze(1).to(torch.float32) / 255
        new_targets = targets[idx][:num_data]

        return torch.utils.data.TensorDataset(new_data, new_targets)

    num_train_data = 2048
    num_classes = 3
    train_tensors = datasets.MNIST(
        "data/",
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    test_tensors = datasets.MNIST(
        "data/",
        train=False,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    train_data = subsample(
        train_tensors.data, train_tensors.targets, num_train_data, num_classes
    )
    test_data = subsample(
        test_tensors.data, test_tensors.targets, num_train_data, num_classes
    )

    mnist_train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True
    )
    mnist_test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False
    )

    # Define prior distribution
    M = args.latent_dim

    def new_encoder():
        encoder_net = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.Softmax(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.Softmax(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.Flatten(),
            nn.Linear(512, 2 * M),
        )
        return encoder_net

    def new_decoder():
        decoder_net = nn.Sequential(
            nn.Linear(M, 512),
            nn.Unflatten(-1, (32, 4, 4)),
            nn.Softmax(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1, output_padding=0),
            nn.Softmax(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.Softmax(),
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
        )
        return decoder_net

    # Choose mode to run
    if args.mode == "train":

        experiments_folder = args.experiment_folder
        os.makedirs(f"{experiments_folder}", exist_ok=True)
        model = VAE(
            GaussianPrior(M),
            GaussianDecoder(new_decoder()),
            GaussianEncoder(new_encoder()),
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        train(
            model,
            optimizer,
            mnist_train_loader,
            args.epochs_per_decoder,
            args.device,
        )
        os.makedirs(f"{experiments_folder}", exist_ok=True)

        torch.save(
            model.state_dict(),
            f"{experiments_folder}/model.pt",
        )

    elif args.mode == "sample":
        model = VAE(
            GaussianPrior(M),
            GaussianDecoder(new_decoder()),
            GaussianEncoder(new_encoder()),
        ).to(device)
        model.load_state_dict(torch.load(args.experiment_folder + "/model.pt"))
        model.eval()

        with torch.no_grad():
            samples = (model.sample(64)).cpu()
            save_image(samples.view(64, 1, 28, 28), args.samples)

            data = next(iter(mnist_test_loader))[0].to(device)
            recon = model.decoder(model.encoder(data).mean).mean
            save_image(
                torch.cat([data.cpu(), recon.cpu()], dim=0), "reconstruction_means.png"
            )

    elif args.mode == "eval":
        # Load trained model
        model = VAE(
            GaussianPrior(M),
            GaussianDecoder(new_decoder()),
            GaussianEncoder(new_encoder()),
        ).to(device)
        model.load_state_dict(torch.load(args.experiment_folder + "/model.pt"))
        model.eval()

        elbos = []
        with torch.no_grad():
            for x, y in mnist_test_loader:
                x = x.to(device)
                elbo = model.elbo(x)
                elbos.append(elbo)
        mean_elbo = torch.tensor(elbos).mean()
        print("Print mean test elbo:", mean_elbo)

    # ----------------- Geodesics -----------------

########################################
## New implementation of the plotting ##
########################################

    elif args.mode == "geodesics":

        # Load trained model
        model = VAE(
            GaussianPrior(M),
            GaussianDecoder(new_decoder()),
            GaussianEncoder(new_encoder()),
        ).to(device)
        model.load_state_dict(torch.load(args.experiment_folder + "/model.pt"))
        model.eval()

        # Collect all encoded points and labels
        all_latents = []
        all_labels = []

        with torch.no_grad():
            for x_batch, y_batch in mnist_test_loader:
                x_batch = x_batch.to(device)
                qz = model.encoder(x_batch)
                z_means = qz.mean.cpu()  # (batch_size, 2)
                all_latents.append(z_means)
                all_labels.append(y_batch)

        all_latents = torch.cat(all_latents, dim=0)  # (2048, 2)
        all_labels = torch.cat(all_labels, dim=0)    # (2048,)

        # Compute geodesics
        num_curves = args.num_curves
        num_t = args.num_t  # number of interpolation points
        data_iter = iter(mnist_test_loader)

        geodesics = []
        latent_pairs = [] # Store the pairs of latent points

        for curve_idx in range(num_curves):
            x_batch, _ = next(data_iter)
            x_batch = x_batch.to(device)

            # Sample two images
            x1 = x_batch[0].unsqueeze(0)
            x2 = x_batch[1].unsqueeze(0)

            # Encode to latent means
            with torch.no_grad():
                z1 = model.encoder(x1).mean.squeeze(0)
                z2 = model.encoder(x2).mean.squeeze(0)

            latent_pairs.append((z1, z2))
            
            print(f"Computing geodesic #{curve_idx+1}")
            geodesic = optimize_geodesic(
                model.decoder, z1, z2, num_points=num_t, num_iters=50, lr=1e-2
            )
            geodesics.append(geodesic)
            
        torch.save(latent_pairs, os.path.join(args.experiment_folder, "latent_pairs.pt"))

        # Plot all
        fig, ax = plt.subplots(figsize=(6, 6))

        # Background = std of decoder output
        z1_min, z1_max = all_latents[:, 0].min().item(), all_latents[:, 0].max().item()
        z2_min, z2_max = all_latents[:, 1].min().item(), all_latents[:, 1].max().item()

        # Add padding for visibility
        padding = 0.7
        zlim = {
            'z1': (z1_min - padding, z1_max + padding),
            'z2': (z2_min - padding, z2_max + padding)
        }

        # Plot background with exact data extent
        plot_latent_std_background(
            model.decoder, ax=ax, device=device,
            z1_range=zlim['z1'], z2_range=zlim['z2'])

        # Scatter plot: all encoded latent means are coloured by class
        cmap = plt.get_cmap('tab10')
        for class_id in torch.unique(all_labels):
            idxs = (all_labels == class_id)
            points = all_latents[idxs]
            ax.scatter(points[:, 0], points[:, 1], s=10, alpha=0.6,
                    label=f"Class {class_id.item()}", color=cmap(class_id.item()))

        # Plot geodesics
        colors = ['C0', 'C1']
        for i, geod in enumerate(geodesics):
            plot_geodesic_latents(geod, ax=ax, color=colors[i % len(colors)], label_once=(i != 0))

        # Legend
        line_geod = mlines.Line2D([], [], color='C0', lw=2, label='Pullback geodesic')
        line_straight = mlines.Line2D([], [], color='C0', lw=2, linestyle='--', alpha=0.6, label='Straight line')
        ax.legend(loc='best', fontsize=8, frameon=True)

        ax.set_title("Multiple Geodesics in Latent Space")
        ax.set_aspect('auto')
        plt.tight_layout()
        plt.show()
        plt.savefig("plot_first_part.png")

########################################
## Old implementation of the plotting ##
########################################

        """
    elif args.mode == "geodesics":

        model = VAE(
            GaussianPrior(M),
            GaussianDecoder(new_decoder()),
            GaussianEncoder(new_encoder()),
        ).to(device)
        model.load_state_dict(torch.load(args.experiment_folder + "/model.pt"))
        model.eval()

        # -- Our implementation --
        
        num_curves = args.num_curves
        num_t = args.num_t  # number of interpolation points
        data_iter = iter(mnist_test_loader)

        geodesics = []

        for curve_idx in range(num_curves):
            x_batch, _ = next(data_iter)
            x_batch = x_batch.to(device)

            # Sample two images
            x1 = x_batch[0].unsqueeze(0)
            x2 = x_batch[1].unsqueeze(0)

            # Encode to latent means
            with torch.no_grad():
                qz1 = model.encoder(x1)
                qz2 = model.encoder(x2)
                z1 = qz1.mean.squeeze(0)
                z2 = qz2.mean.squeeze(0)

            # Compute geodesic
            print(f"Computing geodesic #{curve_idx+1}")
            geodesic = optimize_geodesic(
                model.decoder, z1, z2, num_points=num_t, num_iters=50, lr=1e-2
            )
            geodesics.append(geodesic)
            '''fig, ax = plt.subplots(figsize=(6, 6))

            colors = ['C0', 'C1', 'C2']
            for i, geod in enumerate(geodesic):
                plot_geodesic_latents(geod, ax=ax, color=colors[i])
            
            ax.legend()
            plt.show()'''
            # Esto funciona
            # Plot in latent space
            #plot_geodesic_latents(geodesic)

        fig, ax = plt.subplots(figsize=(6, 6))
        colors = ['C0', 'C1']#, 'C2', 'C3', 'C4', 'C5'] # This can be changed

        # Background
        plot_latent_std_background(model.decoder, ax=ax, device=device)

        for i, geod in enumerate(geodesics):
            plot_geodesic_latents(geod, ax=ax, color=colors[i % len(colors)], label_once=(i != 0))
            #plot_geodesic_latents(geod, ax=ax, color=colors[i % len(colors)])

        line_geod = mlines.Line2D([], [], color='C0', lw=2, label='Pullback geodesic')
        line_straight = mlines.Line2D([], [], color='C0', lw=2, linestyle='--', alpha=0.6, label='Straight line')
        ax.legend(handles=[line_geod, line_straight], loc='upper right')
        #ax.legend() # This works
        plt.title("Multiple Geodesics in Latent Space")
        ax.set_aspect('equal')
        plt.tight_layout()
        plt.show()"""