import torch
import torch.nn as nn
import torch.distributions as td
import torch.utils.data
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from geodesics_utilsbis import optimize_geodesic, plot_geodesic_latents, model_average_energy

# [Previous class definitions for GaussianPrior, GaussianEncoder, GaussianDecoder remain unchanged]
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
    Define a Variational Autoencoder (VAE) model with one encoder and multiple decoders.
    """
    def __init__(self, prior, decoders, encoder):
        """
        Parameters:
        prior: [torch.nn.Module]
           The prior distribution over the latent space.
        decoders: [list of torch.nn.Module]
              List of decoder distributions over the data space.
        encoder: [torch.nn.Module]
                The encoder distribution over the latent space.
        """
        super(VAE, self).__init__()
        self.prior = prior
        self.decoders = nn.ModuleList(decoders)  # List of decoders
        self.encoder = encoder
        self.num_decoders = len(decoders)

    def elbo(self, x):
        """
        Compute the ELBO for the given batch of data, averaging over all decoders.
        """
        q = self.encoder(x)
        z = q.rsample()
        elbo_sum = 0.0
        for decoder in self.decoders:
            elbo_sum += (decoder(z).log_prob(x) - q.log_prob(z) + self.prior().log_prob(z)).mean()
        return elbo_sum / self.num_decoders

    def sample(self, n_samples=1, decoder_idx=None):
        """
        Sample from the model using a specific decoder or randomly chosen one.
        """
        z = self.prior().sample(torch.Size([n_samples]))
        if decoder_idx is None:
            decoder_idx = torch.randint(0, self.num_decoders, (1,)).item()
        return self.decoders[decoder_idx](z).sample()

    def forward(self, x):
        """
        Compute the negative ELBO for the given batch of data.
        """
        return -self.elbo(x)

def train(model, optimizer, data_loader, epochs, device):
    """
    Train a VAE model with multiple decoders.
    """
    num_steps = len(data_loader) * epochs
    epoch = 0  # Changed from 20 to 0 to start correctly

    def noise(x, std=0.05):
        eps = std * torch.randn_like(x)
        return torch.clamp(x + eps, min=0.0, max=1.0)

    with tqdm(range(num_steps)) as pbar:
        for step in pbar:
            try:
                x = next(iter(data_loader))[0]
                x = noise(x.to(device))
                optimizer.zero_grad()
                loss = model(x)
                loss.backward()
                optimizer.step()

                if step % 5 == 0:
                    loss = loss.detach().cpu()
                    pbar.set_description(f"epoch={epoch}, step={step}, loss={loss:.1f}")

                if (step + 1) % len(data_loader) == 0:
                    epoch += 1
            except KeyboardInterrupt:
                print(f"Stopping training at epoch {epoch} and loss: {loss:.1f}")
                break

if __name__ == "__main__":
    from torchvision import datasets, transforms
    from torchvision.utils import save_image
    import argparse

    # [Argument parsing remains unchanged]

    parser = argparse.ArgumentParser()
    # [Previous parser arguments remain unchanged]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        type=str,
        default="train",
        choices=["train", "sample", "eval", "geodesics", "cov_plot"],
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

    parser.add_argument(
        "--num-iters",  # number of points along the curve
        type=int,
        default=50,
        metavar="N",
        help="number of iterations to optimize the geodesic (default: %(default)s)",
    )

    parser.add_argument(
        "--lr",  # number of points along the curve
        type=float,
        default=1e-3,
        help="learning rate to use for optimization of the geodesic (default: %(default)s)",
    )

    args = parser.parse_args()
    device = args.device

    # Load MNIST data (unchanged)
    def subsample(data, targets, num_data, num_classes):
        idx = targets < num_classes
        new_data = data[idx][:num_data].unsqueeze(1).to(torch.float32) / 255
        new_targets = targets[idx][:num_data]
        return torch.utils.data.TensorDataset(new_data, new_targets)

    num_train_data = 2048
    num_classes = 3
    train_tensors = datasets.MNIST("data/", train=True, download=True, transform=transforms.ToTensor())
    test_tensors = datasets.MNIST("data/", train=False, download=True, transform=transforms.ToTensor())
    train_data = subsample(train_tensors.data, train_tensors.targets, num_train_data, num_classes)
    test_data = subsample(test_tensors.data, test_tensors.targets, num_train_data, num_classes)

    mnist_train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    mnist_test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    # Define prior and network architectures
    M = args.latent_dim

    def new_encoder():
        return nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.Softmax(dim=1),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.Softmax(dim=1),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.Flatten(),
            nn.Linear(512, 2 * M),
        )

    def new_decoder():
        return nn.Sequential(
            nn.Linear(M, 512),
            nn.Unflatten(-1, (32, 4, 4)),
            nn.Softmax(dim=1),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1, output_padding=0),
            nn.Softmax(dim=1),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.Softmax(dim=1),
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
        )

    # Mode: Train
    if args.mode == "train":
        experiments_folder = args.experiment_folder
        os.makedirs(f"{experiments_folder}", exist_ok=True)

        # Create a single VAE with multiple decoders
        decoders = [GaussianDecoder(new_decoder()) for _ in range(args.num_decoders)]
        model = VAE(GaussianPrior(M), decoders, GaussianEncoder(new_encoder())).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        print(f"Training VAE with {args.num_decoders} decoders")
        train(model, optimizer, mnist_train_loader, args.epochs_per_decoder, device)
        torch.save(model.state_dict(), f"{experiments_folder}/vae_with_{args.num_decoders}_decoders.pt")

    # Mode: Geodesics
    elif args.mode == "geodesics":
        import matplotlib.lines as mlines

        # Load the single VAE with multiple decoders
        decoders = [GaussianDecoder(new_decoder()) for _ in range(args.num_decoders)]
        model = VAE(GaussianPrior(M), decoders, GaussianEncoder(new_encoder())).to(device)
        model.load_state_dict(torch.load(f"{args.experiment_folder}/vae_with_{args.num_decoders}_decoders.pt"))
        model.eval()

        # Use the model's decoders list directly
        decoders = model.decoders

        # Load latent pairs
        latent_pairs = torch.load(os.path.join(args.experiment_folder, "latent_pairs.pt"))
        geodesics = []

        # Compute geodesics for each pair using the single VAE's decoders
        for curve_idx, (z1, z2) in enumerate(latent_pairs):
            print(f"Computing geodesic #{curve_idx+1}/{len(latent_pairs)}")
            geodesic = optimize_geodesic(
                decoders=decoders,
                z_start=z1,
                z_end=z2,
                num_points=args.num_t,
                num_iters=args.num_iters,
                lr=args.lr,
                energy_fn=lambda c: model_average_energy(c, decoders, num_samples=50),
                convergence_threshold=1e-3,
                window_size=10
            )
            geodesics.append(geodesic)

        # Plotting (unchanged from original)
        fig, ax = plt.subplots(figsize=(6, 6))
        all_latents = torch.stack([z for pair in latent_pairs for z in pair])
        z1_min, z1_max = all_latents[:, 0].min().item(), all_latents[:, 0].max().item()
        z2_min, z2_max = all_latents[:, 1].min().item(), all_latents[:, 1].max().item()
        padding = 0.7
        zlim = {'z1': (z1_min - padding, z1_max + padding), 'z2': (z2_min - padding, z2_max + padding)}

        # from geodesics_utilsbis import plot_latent_std_background
        # plot_latent_std_background(decoders[0], ax=ax, device=device, z1_range=zlim['z1'], z2_range=zlim['z2'])
        from geodesics_utilsbis import plot_latent_std_background_across_decoders
        plot_latent_std_background_across_decoders(decoders, device=device, ax=ax)

        ax.scatter(all_latents[:, 0].cpu(), all_latents[:, 1].cpu(), s=10, alpha=0.6, color='gray', label='Latent Points')

        colors = ['C0', 'C1']
        for i, geod in enumerate(geodesics):
            geod_cpu = geod.cpu()
            plot_geodesic_latents(geod_cpu, ax=ax, color=colors[i % len(colors)], label_once=(i != 0))

        line_geod = mlines.Line2D([], [], color='C0', lw=2, label='Pullback geodesic')
        line_straight = mlines.Line2D([], [], color='C0', lw=2, linestyle='--', alpha=0.6, label='Straight line')
        ax.legend(loc='best', fontsize=8, frameon=True)
        ax.set_title("Multiple Geodesics in Latent Space (Single VAE, Multiple Decoders)")
        ax.set_aspect('auto')
        plt.tight_layout()
        plt.savefig("geodesic_plot_single_vae.png")
        plt.show()

    # [Other modes like "sample", "eval", "cov_plot" can be adapted similarly if needed]