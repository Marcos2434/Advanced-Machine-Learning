# Code for DTU course 02460 (Advanced Machine Learning Spring) by Jes Frellsen, 2024
# Version 1.0 (2024-01-27)
# Inspiration is taken from:
# - https://github.com/jmtomczak/intro_dgm/blob/main/vaes/vae_example.ipynb
# - https://github.com/kampta/pytorch-distributions/blob/master/gaussian_vae.py
#
# Significant extension by Søren Hauberg, 2024

# To run it
# python Project2bis_ensemble_vae.py geodesics --device cuda --num-curves 25 --num-t 15 --num-decoders 10
# To test it
# python Project2bis_ensemble_vae.py geodesics --device cuda --num-curves 2 --num-t 3 --num-decoders 2
# To save an image:
# save_image(torch.cat([data.cpu(), recon.cpu()], dim=0), "something.png")
# TODO: Add all the 2048 points 


# To run it
# python Project2bis_ensemble_vae.py cov_plot --device cuda --num-curves 25 --num-t 15 --num-decoders 10
# To test it
# python Project2bis_ensemble_vae.py cov_plot --device cuda --num-curves 2 --num-t 3 --num-decoders 2
# To save an image:
# save_image(torch.cat([data.cpu(), recon.cpu()], dim=0), "something.png")
# TODO: Add all the 2048 points 

import torch
import torch.nn as nn
import torch.distributions as td
import torch.utils.data
from tqdm import tqdm
from copy import deepcopy
import os
import matplotlib.pyplot as plt
from geodesics_utilsbis import optimize_geodesic, plot_geodesic_latents,plot_decoded_images,plot_latent_std_background,compute_curve_energies, compute_energy, compute_length



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
            nn.Softmax(dim=1),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.Softmax(dim=1),
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
        return decoder_net

    # Choose mode to run
    if args.mode == "train":

        experiments_folder = args.experiment_folder
        os.makedirs(f"{experiments_folder}", exist_ok=True)

        for i in range(args.num_decoders):
            print(f"\nTraining decoder #{i+1}/{args.num_decoders}")
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

            torch.save(model.state_dict(), f"{experiments_folder}/model_{args.num_decoders}decoders_{i}.pt")

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

    # --------------- Geodesics -------------------
    
    elif args.mode == "geodesics":
        
        import random
        
        decoders = []
        for i in range(args.num_decoders):
            model = VAE(
                GaussianPrior(M),
                GaussianDecoder(new_decoder()),
                GaussianEncoder(new_encoder())
            ).to(device)
            model.load_state_dict(torch.load(f"{args.experiment_folder}/model_{args.num_decoders}decoders_{i}.pt"))
            model.eval()
            decoders.append(model.decoder)

        
        N = args.num_t
        t = torch.linspace(0, 1, N+1)
        def Epsilon(c, decoders):
            fl = random.choice(decoders)
            fk = random.choice(decoders)
            
            energy = 0.0
            for i in range(N+1):
                f1 = fl(c[i].unsqueeze(0)).mean.view(-1)
                f2 = fk(c[i+1].unsqueeze(0)).mean.view(-1)
                energy += torch.norm(f1 - f2)**2
            return energy / N
        
        
        latent_pairs = torch.load(os.path.join(args.experiment_folder, "latent_pairs.pt"))
        geodesics = []
        
        for curve_idx, (z1, z2) in enumerate(latent_pairs):
            print(f"Computing ensemble geodesic #{curve_idx+1}")
            geodesic = optimize_geodesic(
                decoders=decoders, z_start=z1, z_end=z2, num_points=args.num_t, num_iters=50, lr=1e-2, energy_fn=Epsilon
            )
            geodesics.append(geodesic)
            
        fig, ax = plt.subplots()
        # scatter all latent points
        ax.scatter(latent_pairs[:, 0], latent_pairs[:, 1], alpha=0.3)

        # for each pair
        for z1, z2 in latent_pairs:
            geod = optimize_geodesic(z1, z2, decoders=decoders, energy_fn=Epsilon)
            plot_geodesic_latents(geod, ax=ax)

        plt.title("Latent space with ensemble geodesics")
        plt.show()

    elif args.mode == "cov_plot":
        import random

        decoder_counts = [1, 2, 3, 5, 10]  # Try different ensemble sizes
        cov_geo_list = []
        cov_euc_list = []

        # Load all decoders once
        all_decoders = []
        for i in range(args.num_deocders):
            model = VAE(
                GaussianPrior(M),
                GaussianDecoder(new_decoder()),
                GaussianEncoder(new_encoder()),
            ).to(device)
            model.load_state_dict(torch.load(f"{args.experiment_folder}/model_{args.num_deocders}decoders_{i}.pt"))
            model.eval()
            all_decoders.append(model.decoder)

        # Load same latent pairs (e.g. 10)
        latent_pairs = torch.load(os.path.join(args.experiment_folder, "latent_pairs.pt"))
        latent_pairs = latent_pairs[:10]

        for num_dec in decoder_counts:
            dec_subset = all_decoders[:num_dec] # Select first num_dec decoders from the full ensemble
            
            # collect energy values and Euclidean distances for all pairs
            geo_energies = []
            euc_dists = []

            for z1, z2 in latent_pairs:
                # Compute geodesic energies
                curve_set = [] 
                
                # find geodesic for latent pair for each decoder
                for dec in dec_subset:
                    geod = optimize_geodesic(dec, z1, z2, num_points=args.num_t, num_iters=30, lr=1e-2)
                    curve_set.append(geod)
                curve_energies = [compute_energy(dec, geod).item() for dec, geod in zip(dec_subset, curve_set)]
                geo_energies.append(curve_energies)

                # Euclidean norm (same across decoders)
                euc = torch.norm(z2 - z1).item()
                euc_dists.append(euc)

            # CoV for geodesics
            geo_tensor = torch.tensor(geo_energies)  # shape: (num_pairs, num_dec)
            cov_geo = (geo_tensor.std(dim=1) / geo_tensor.mean(dim=1)).mean().item()
            cov_geo_list.append(cov_geo)

            # CoV for Euclidean = 0 (no variation), but still include for symmetry
            cov_euc_list.append(0.0)

            print(f"{num_dec} decoders -> Geodesic CoV: {cov_geo:.4f}")

        # Plot
        plt.plot(decoder_counts, cov_geo_list, marker='o', label='Geodesic CoV')
        plt.plot(decoder_counts, cov_euc_list, marker='x', label='Euclidean CoV')
        plt.xlabel("Number of decoders")
        plt.ylabel("Average CoV across 10 pairs")
        plt.title("Coefficient of Variation vs Ensemble Size")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    
    # elif args.mode == "geodesics":

    #     print("Loading ensemble of VAE models...")
    #     models = []
    #     for i in range(args.num_decoders):
    #         model = VAE(
    #             GaussianPrior(M),
    #             GaussianDecoder(new_decoder()),
    #             GaussianEncoder(new_encoder()),
    #         ).to(device)
    #         model.load_state_dict(torch.load(f"{args.experiment_folder}/model_{args.num_decoders}decoders_{i}.pt"))
    #         model.eval()
    #         models.append(model)

    #     print("Computing geodesics and energies...")

    #     num_curves = args.num_curves
    #     num_t = args.num_t
    #     data_iter = iter(mnist_test_loader)

    #     geodesics_all = []  # shape: (num_curves, num_decoders)
    #     z_pairs = []

    #     for curve_idx in range(num_curves):
    #         print(f"[{curve_idx+1}/{num_curves}] Computing geodesics for curve {curve_idx+1}...")

    #         x_batch, _ = next(data_iter)
    #         x_batch = x_batch.to(device)
    #         x1 = x_batch[0].unsqueeze(0)
    #         x2 = x_batch[1].unsqueeze(0)

    #         curve_set = []
    #         for model in models:
    #             encoder = model.encoder
    #             decoder = model.decoder

    #             with torch.no_grad():
    #                 z1 = encoder(x1).mean.squeeze(0)
    #                 z2 = encoder(x2).mean.squeeze(0)

    #             geod = optimize_geodesic(decoder, z1, z2, num_points=num_t, num_iters=50, lr=1e-2)
    #             curve_set.append(geod)

    #         geodesics_all.append(curve_set)
    #         z_pairs.append((z1, z2))  # Save z1/z2 for later Euclidean distance

    #     # Compute geodesic energies
    #     energies = []
    #     for curve_set in geodesics_all:
    #         curve_energies = [compute_energy(decoder, curve).item()
    #                         for decoder, curve in zip([m.decoder for m in models], curve_set)]
    #         energies.append(curve_energies)

    #     energies_tensor = torch.tensor(energies)  # shape: (num_curves, num_decoders)
    #     avg_energies = energies_tensor.mean(dim=1)
    #     std_energies = energies_tensor.std(dim=1)
    #     cov_geodesic = (std_energies / avg_energies).mean().item()
    #     mean_energy = avg_energies.mean().item()

    #     print(f"Average model-averaged geodesic energy: {mean_energy:.4f}")
    #     print(f"Geodesic CoV: {cov_geodesic:.4f}")

    #     # Compute geodesic lengths
    #     lengths = []
    #     for curve_set in geodesics_all:
    #         curve_lengths = [compute_length(decoder, curve)
    #                         for decoder, curve in zip([m.decoder for m in models], curve_set)]
    #         lengths.append(curve_lengths)

    #     lengths_tensor = torch.tensor(lengths)  # shape: (num_curves, num_decoders)
    #     avg_lengths = lengths_tensor.mean(dim=1)
    #     std_lengths = lengths_tensor.std(dim=1)
    #     cov_length = (std_lengths / avg_lengths).mean().item()
    #     mean_length = avg_lengths.mean().item()

    #     print(f"Average Geodesic Length: {mean_length:.4f}")
    #     print(f"Geodesic Length CoV: {cov_length:.4f}")

    #     # Euclidean distances using z_pairs from earlier
    #     euclidean_dists = [torch.norm(z2 - z1).item() for z1, z2 in z_pairs]
    #     euclidean_dists = torch.tensor(euclidean_dists)
    #     mean_euclidean = euclidean_dists.mean()
    #     std_euclidean = euclidean_dists.std()
    #     cov_euclidean = (std_euclidean / mean_euclidean).item()

    #     print(f"Average Euclidean Distance: {mean_euclidean:.4f}")
    #     print(f"Euclidean CoV: {cov_euclidean:.4f}")

"""    elif args.mode == "geodesics":

        print("Loading ensemble of VAE models...")
        models = []
        for i in range(args.num_decoders):
            model = VAE(
                GaussianPrior(M),
                GaussianDecoder(new_decoder()),
                GaussianEncoder(new_encoder()),
            ).to(device)
            model.load_state_dict(torch.load(f"{args.experiment_folder}/model_{args.num_decoders}decoders_{i}.pt"))
            model.eval()
            models.append(model)

        print("Computing geodesics and energies...")

        num_curves = args.num_curves
        num_t = args.num_t
        data_iter = iter(mnist_test_loader)

        geodesics_all = []  # shape: (num_curves, num_decoders)
        z_pairs = []        # Save z1, z2 pairs per decoder (optional)
        
        for curve_idx in range(num_curves):
            print(f"[{curve_idx+1}/{num_curves}] Computing geodesics for curve {curve_idx+1}...")

            x_batch, _ = next(data_iter)
            x_batch = x_batch.to(device)
            x1 = x_batch[0].unsqueeze(0)
            x2 = x_batch[1].unsqueeze(0)

            curve_set = []
            for model in models:
                encoder = model.encoder
                decoder = model.decoder

                with torch.no_grad():
                    z1 = encoder(x1).mean.squeeze(0)
                    z2 = encoder(x2).mean.squeeze(0)

                geod = optimize_geodesic(decoder, z1, z2, num_points=num_t, num_iters=50, lr=1e-2)
                curve_set.append(geod)

            geodesics_all.append(curve_set)

        # Compute geodesic energies
        energies = []
        for curve_set in geodesics_all:
            curve_energies = [compute_energy(decoder, curve).item()
                            for decoder, curve in zip([m.decoder for m in models], curve_set)]
            energies.append(curve_energies)

        energies_tensor = torch.tensor(energies)  # shape: (num_curves, num_decoders)
        avg_energies = energies_tensor.mean(dim=1)
        std_energies = energies_tensor.std(dim=1)
        cov_geodesic = (std_energies / avg_energies).mean().item()
        mean_energy = avg_energies.mean().item()

        print(f"Average model-averaged geodesic energy: {mean_energy:.4f}")
        print(f"Geodesic CoV: {cov_geodesic:.4f}")

        # (Optional) Euclidean distance based on first model’s encoder
        euclidean_dists = []
        for curve_idx in range(num_curves):
            x_batch, _ = next(iter(mnist_test_loader))
            x_batch = x_batch.to(device)
            x1 = x_batch[0].unsqueeze(0)
            x2 = x_batch[1].unsqueeze(0)

            with torch.no_grad():
                z1 = models[0].encoder(x1).mean.squeeze(0)
                z2 = models[0].encoder(x2).mean.squeeze(0)
            euclidean_dists.append(torch.norm(z2 - z1).item())

        euclidean_dists = torch.tensor(euclidean_dists)
        mean_euclidean = euclidean_dists.mean()
        std_euclidean = euclidean_dists.std()
        cov_euclidean = (std_euclidean / mean_euclidean).item()

        print(f"Average Euclidean Distance: {mean_euclidean:.4f}")
        print(f"Euclidean CoV: {cov_euclidean:.4f}")"""

    # This kind of works
"""print("Loading ensemble of decoders...")
        decoders = []
        for i in range(args.num_decoders):
            model = VAE(
                GaussianPrior(M),
                GaussianDecoder(new_decoder()),
                GaussianEncoder(new_encoder()),
            ).to(device)
            model.load_state_dict(torch.load(f"{args.experiment_folder}/model_{args.num_decoders}decoders_{i}.pt")) # Change here !!
            model.eval()
            decoders.append(model.decoder)

        print("Computing geodesics and energies...")

        num_curves = args.num_curves
        num_t = args.num_t
        data_iter = iter(mnist_test_loader)

        geodesics_all = []  # shape: (num_curves, num_decoders)
        z_pairs = []        # Save z1, z2 pairs for Euclidean

        for curve_idx in range(num_curves):
            x_batch, _ = next(data_iter)
            x_batch = x_batch.to(device)
            x1 = x_batch[0].unsqueeze(0)
            x2 = x_batch[1].unsqueeze(0)

            with torch.no_grad():
                z1 = model.encoder(x1).mean.squeeze(0)
                z2 = model.encoder(x2).mean.squeeze(0)
            z_pairs.append((z1, z2))

            curve_set = []
            for decoder in decoders:
                geod = optimize_geodesic(decoder, z1, z2, num_points=num_t, num_iters=50, lr=1e-2)
                curve_set.append(geod)
            geodesics_all.append(curve_set)

        # Compute geodesic energies
        energies = []
        for curve_set in geodesics_all:
            curve_energies = [compute_energy(decoder, curve).item()
                            for decoder, curve in zip(decoders, curve_set)]
            energies.append(curve_energies)

        energies_tensor = torch.tensor(energies)  # shape: (num_curves, num_decoders)
        avg_energies = energies_tensor.mean(dim=1)
        std_energies = energies_tensor.std(dim=1)
        cov_geodesic = (std_energies / avg_energies).mean().item()
        mean_energy = avg_energies.mean().item()

        print(f"Average model-averaged geodesic energy: {mean_energy:.4f}")
        print(f"Geodesic CoV: {cov_geodesic:.4f}")

        # Compute Euclidean distances and CoV
        euclidean_dists = [torch.norm(z2 - z1).item() for z1, z2 in z_pairs]
        euclidean_dists = torch.tensor(euclidean_dists)
        mean_euclidean = euclidean_dists.mean()
        std_euclidean = euclidean_dists.std()
        cov_euclidean = (std_euclidean / mean_euclidean).item()

        print(f"Average Euclidean Distance: {mean_euclidean:.4f}")
        print(f"Euclidean CoV: {cov_euclidean:.4f}")"""
