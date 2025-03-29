import torch
import torch.nn as nn
import torch.distributions as td
import torch.utils.data
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
from geodesics_utilsbis import optimize_geodesic, plot_geodesic_latents, compute_energy, model_average_energy, compute_length

class GaussianPrior(nn.Module):
    def __init__(self, M):
        super().__init__()
        self.M = M
        self.mean = nn.Parameter(torch.zeros(M), requires_grad=False)
        self.logvar = nn.Parameter(torch.zeros(M), requires_grad=False)

    def forward(self):
        return td.Independent(td.Normal(loc=self.mean, scale=torch.exp(0.5 * self.logvar)), 1)

class GaussianDecoder(nn.Module):
    def __init__(self, decoder_net):
        super().__init__()
        self.net = decoder_net
        self.logvar = nn.Parameter(torch.tensor(0.0))

    def forward(self, z):
        mean = self.net(z)
        return td.Independent(
            td.Normal(loc=mean, scale=torch.exp(0.5 * self.logvar).expand_as(mean)), 3
        )

class GaussianEncoder(nn.Module):
    def __init__(self, encoder_net):
        super().__init__()
        self.net = encoder_net

    def forward(self, x):
        mean_logvar = self.net(x)
        mean, logvar = mean_logvar[:, : mean_logvar.shape[1] // 2], mean_logvar[:, mean_logvar.shape[1] // 2 :]
        return td.Independent(td.Normal(loc=mean, scale=torch.exp(0.5 * logvar)), 1)

class VAE(nn.Module):
    def __init__(self, prior, decoders, encoder):
        super().__init__()
        self.prior = prior
        self.decoders = nn.ModuleList(decoders)  # List of decoders
        self.encoder = encoder
        self.num_decoders = len(decoders)

    def elbo(self, x):
        q = self.encoder(x)
        z = q.rsample()
        elbo_sum = 0.0
        for decoder in self.decoders:
            elbo_sum += (decoder(z).log_prob(x) - q.log_prob(z) + self.prior().log_prob(z)).mean()
        return elbo_sum / self.num_decoders

    def sample(self, n_samples=1, decoder_idx=None):
        z = self.prior().sample(torch.Size([n_samples]))
        if decoder_idx is None:
            decoder_idx = torch.randint(0, self.num_decoders, (1,)).item()
        return self.decoders[decoder_idx](z).sample()

    def forward(self, x):
        return -self.elbo(x)

def train(model, optimizer, data_loader, epochs, device):
    num_steps = len(data_loader) * epochs
    epoch = 0

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

    # Argument parsing
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
        help="batch size for training (default: %(default)s)",
    )
    parser.add_argument(
        "--epochs-per-decoder",
        type=int,
        default=50,
        help="number of training epochs per decoder (default: %(default)s)",
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=2,
        help="dimension of latent variable (default: %(default)s)",
    )
    parser.add_argument(
        "--num-decoders",
        type=int,
        default=1,
        help="number of decoders in the ensemble (default: %(default)s)",
    )
    parser.add_argument(
        "--num-reruns",
        type=int,
        default=10,
        help="number of reruns (default: %(default)s)",
    )
    parser.add_argument(
        "--num-curves",
        type=int,
        default=25,
        help="number of geodesics to plot (default: %(default)s)",
    )
    parser.add_argument(
        "--num-t",
        type=int,
        default=15,
        help="number of points along the curve (default: %(default)s)",
    )
    parser.add_argument(
        "--num-iters",
        type=int,
        default=50,
        help="number of iterations to optimize the geodesic (default: %(default)s)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="learning rate for optimization (default: %(default)s)",
    )

    args = parser.parse_args()
    print("# Options")
    for key, value in sorted(vars(args).items()):
        print(key, "=", value)
    device = args.device

    # Load MNIST data
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

        # Create a VAE with the specified number of decoders
        decoders = [GaussianDecoder(new_decoder()) for _ in range(args.num_decoders)]
        model = VAE(GaussianPrior(M), decoders, GaussianEncoder(new_encoder())).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        print(f"Training VAE with {args.num_decoders} decoders")
        train(model, optimizer, mnist_train_loader, args.epochs_per_decoder, device)
        torch.save(model.state_dict(), f"{experiments_folder}/vae_with_{args.num_decoders}_decoders.pt")

        # For Part A: Also train a single-decoder VAE if not already trained
        single_decoder_path = f"{experiments_folder}/vae_single_decoder.pt"
        if not os.path.exists(single_decoder_path):
            print("Training single-decoder VAE for Part A")
            single_decoder_model = VAE(
                GaussianPrior(M),
                [GaussianDecoder(new_decoder())],
                GaussianEncoder(new_encoder())
            ).to(device)
            optimizer = torch.optim.Adam(single_decoder_model.parameters(), lr=args.lr)
            train(single_decoder_model, optimizer, mnist_train_loader, args.epochs_per_decoder, device)
            torch.save(single_decoder_model.state_dict(), single_decoder_path)

    elif args.mode == "geodesics":
        # Load the single-decoder VAE for Part A
        single_decoder_path = f"{args.experiment_folder}/vae_single_decoder.pt"
        single_decoder_model = VAE(
            GaussianPrior(M),
            [GaussianDecoder(new_decoder())],
            GaussianEncoder(new_encoder())
        ).to(device)
        single_decoder_model.load_state_dict(torch.load(single_decoder_path))
        single_decoder_model.eval()

        # Load the ensemble VAE for Part B
        ensemble_path = f"{args.experiment_folder}/vae_with_{args.num_decoders}_decoders.pt"
        ensemble_model = VAE(
            GaussianPrior(M),
            [GaussianDecoder(new_decoder()) for _ in range(args.num_decoders)],
            GaussianEncoder(new_encoder())
        ).to(device)
        ensemble_model.load_state_dict(torch.load(ensemble_path))
        ensemble_model.eval()

        # Step 1: Generate 25 random latent pairs using the single-decoder VAE
        latent_pairs = []
        data_iter = iter(mnist_test_loader)
        for _ in range(args.num_curves):
            x_batch, _ = next(data_iter)
            x_batch = x_batch.to(device)
            x1 = x_batch[0].unsqueeze(0)  # Shape: (1, 1, 28, 28)
            x2 = x_batch[1].unsqueeze(0)
            with torch.no_grad():
                z1 = single_decoder_model.encoder(x1).mean.squeeze(0)  # Shape: (2,)
                z2 = single_decoder_model.encoder(x2).mean.squeeze(0)
            latent_pairs.append((z1, z2))
        torch.save(latent_pairs, os.path.join(args.experiment_folder, "latent_pairs.pt"))

        # Step 2: Compute geodesics for Part A (single decoder)
        single_decoder = single_decoder_model.decoders[0]  # Single decoder
        single_geodesics = []
        for curve_idx, (z1, z2) in enumerate(latent_pairs):
            print(f"Computing geodesic #{curve_idx+1}/{len(latent_pairs)} (Single Decoder)")
            geodesic = optimize_geodesic(
                decoders=[single_decoder],  # Single decoder
                z_start=z1,
                z_end=z2,
                num_points=args.num_t,
                num_iters=args.num_iters,
                lr=args.lr,
                energy_fn=lambda c: compute_energy(single_decoder, c),
                convergence_threshold=1e-3,
                window_size=10
            )
            single_geodesics.append(geodesic)

        # Step 3: Compute geodesics for Part B (ensemble decoder with model_average_energy)
        ensemble_decoders = ensemble_model.decoders  # List of decoders
        ensemble_geodesics = []
        for curve_idx, (z1, z2) in enumerate(latent_pairs):
            print(f"Computing geodesic #{curve_idx+1}/{len(latent_pairs)} (Ensemble Decoder)")
            geodesic = optimize_geodesic(
                decoders=ensemble_decoders,
                z_start=z1,
                z_end=z2,
                num_points=args.num_t,
                num_iters=args.num_iters,
                lr=args.lr,
                energy_fn=lambda c: model_average_energy(c, ensemble_decoders, num_samples=50),
                convergence_threshold=1e-3,
                window_size=10
            )
            ensemble_geodesics.append(geodesic)

        # Step 4: Plotting for Part A (Single Decoder)
        fig, ax = plt.subplots(figsize=(6, 6))
        all_latents = torch.stack([z for pair in latent_pairs for z in pair])
        z1_min, z1_max = all_latents[:, 0].min().item(), all_latents[:, 0].max().item()
        z2_min, z2_max = all_latents[:, 1].min().item(), all_latents[:, 1].max().item()
        padding = 0.7
        zlim = {'z1': (z1_min - padding, z1_max + padding), 'z2': (z2_min - padding, z2_max + padding)}

        ax.scatter(all_latents[:, 0].cpu(), all_latents[:, 1].cpu(), s=10, alpha=0.6, color='gray', label='Latent Points')

        colors = ['C0', 'C1']
        for i, geod in enumerate(single_geodesics):
            geod_cpu = geod.cpu()
            plot_geodesic_latents(geod_cpu, ax=ax, color=colors[i % len(colors)], label_once=(i != 0))

        line_geod = mlines.Line2D([], [], color='C0', lw=2, label='Pullback geodesic')
        line_straight = mlines.Line2D([], [], color='C0', lw=2, linestyle='--', alpha=0.6, label='Straight line')
        ax.legend(loc='best', fontsize=8, frameon=True)
        ax.set_title("Geodesics in Latent Space (Single Decoder VAE)")
        ax.set_aspect('auto')
        plt.tight_layout()
        plt.savefig("geodesic_plot_single_decoder.png")
        plt.show()

        # Step 5: Plotting for Part B (Ensemble Decoder)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(all_latents[:, 0].cpu(), all_latents[:, 1].cpu(), s=10, alpha=0.6, color='gray', label='Latent Points')

        for i, geod in enumerate(ensemble_geodesics):
            geod_cpu = geod.cpu()
            plot_geodesic_latents(geod_cpu, ax=ax, color=colors[i % len(colors)], label_once=(i != 0))

        line_geod = mlines.Line2D([], [], color='C0', lw=2, label='Pullback geodesic')
        line_straight = mlines.Line2D([], [], color='C0', lw=2, linestyle='--', alpha=0.6, label='Straight line')
        ax.legend(loc='best', fontsize=8, frameon=True)
        ax.set_title(f"Geodesics in Latent Space (Ensemble VAE with {args.num_decoders} Decoders)")
        ax.set_aspect('auto')
        plt.tight_layout()
        plt.savefig("geodesic_plot_ensemble_decoder.png")
        plt.show()

    elif args.mode == "cov_plot":
        # Define ensemble sizes to evaluate
        max_decoders = 10  # Maximum number of decoders
        decoder_counts = list(range(1, max_decoders + 1))  # [1, 2, ..., 10]
        M = 10  # Number of VAEs
        num_pairs = 10  # Number of test point pairs

        cov_geo_list = []  # Geodesic CoV for each ensemble size
        cov_euc_list = []  # Euclidean CoV for each ensemble size

        # Step 1: Train M=10 VAEs with max_decoders (10) decoders if not already trained
        experiments_folder = f"{args.experiment_folder}/ensemble_max_decoders"
        os.makedirs(experiments_folder, exist_ok=True)

        # Train M VAEs with max_decoders
        for m in range(M):
            model_path = f"{experiments_folder}/vae_{m}.pt"
            if not os.path.exists(model_path):
                print(f"Training VAE #{m+1}/{M} with {max_decoders} decoders")
                decoders = [GaussianDecoder(new_decoder()) for _ in range(max_decoders)]
                model = VAE(GaussianPrior(M), decoders, GaussianEncoder(new_encoder())).to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
                train(model, optimizer, mnist_train_loader, args.epochs_per_decoder, device)
                torch.save(model.state_dict(), model_path)

        # Step 2: Load the M VAEs and store their encoders and decoders
        models = []
        for m in range(M):
            model = VAE(
                GaussianPrior(M),
                [GaussianDecoder(new_decoder()) for _ in range(max_decoders)],
                GaussianEncoder(new_encoder())
            ).to(device)
            model.load_state_dict(torch.load(f"{experiments_folder}/vae_{m}.pt"))
            model.eval()
            models.append(model)

        # Step 3: Select test point pairs and compute latent representations
        # Since all VAEs share the same encoder architecture, we can use any encoder to get consistent latent points
        data_iter = iter(mnist_test_loader)
        test_pairs = []
        latent_pairs = []
        for _ in range(num_pairs):
            x_batch, _ = next(data_iter)
            x_batch = x_batch.to(device)
            x1 = x_batch[0].unsqueeze(0)  # Shape: (1, 1, 28, 28)
            x2 = x_batch[1].unsqueeze(0)
            test_pairs.append((x1, x2))
            with torch.no_grad():
                z1 = models[0].encoder(x1).mean.squeeze(0)  # Shape: (2,)
                z2 = models[0].encoder(x2).mean.squeeze(0)
            latent_pairs.append((z1, z2))

        # Step 4: Compute CoV for each number of decoders
        for num_dec in decoder_counts:
            print(f"\n--- Analyzing Ensemble Size {num_dec} Decoders ---")

            geo_lengths_per_pair = []  # List of lists: [pair1_lengths, pair2_lengths, ...]
            euc_dists_per_pair = []    # List of lists: [pair1_dists, pair2_dists, ...]

            # For each pair of test points
            for pair_idx, (z1, z2) in enumerate(latent_pairs):
                print(f"Processing pair {pair_idx+1}/{num_pairs}")
                geo_lengths = []  # Geodesic lengths for this pair across M VAEs
                euc_dists = []    # Euclidean distances for this pair across M VAEs

                # Compute distances using each VAE
                for model_idx, model in enumerate(models):
                    # Euclidean distance (should be the same for all models since encoders are consistent)
                    euc_dist = torch.norm(z2 - z1).item()
                    euc_dists.append(euc_dist)

                    # Geodesic distance (using the first num_dec decoders)
                    decoders = model.decoders[:num_dec]  # Subset of decoders
                    geod = optimize_geodesic(
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
                    # Compute average geodesic length over the subset of decoders
                    geo_length = sum(compute_length(dec, geod) for dec in decoders) / len(decoders)
                    geo_lengths.append(geo_length)

                    print(f"  VAE {model_idx+1}: Euclidean={euc_dist:.4f}, Geodesic={geo_length:.4f}")

                geo_lengths_per_pair.append(geo_lengths)
                euc_dists_per_pair.append(euc_dists)

            # Compute CoV for this ensemble size
            geo_lengths_tensor = torch.tensor(geo_lengths_per_pair)  # Shape: (num_pairs, M)
            euc_dists_tensor = torch.tensor(euc_dists_per_pair)      # Shape: (num_pairs, M)

            # CoV per pair, then average
            geo_cov = (geo_lengths_tensor.std(dim=1) / geo_lengths_tensor.mean(dim=1)).mean().item()
            euc_cov = (euc_dists_tensor.std(dim=1) / euc_dists_tensor.mean(dim=1)).mean().item()

            cov_geo_list.append(geo_cov)
            cov_euc_list.append(euc_cov)
            print(f"Ensemble size {num_dec}: Geodesic CoV={geo_cov:.4f}, Euclidean CoV={euc_cov:.4f}")

        # Step 5: Plot the results
        plt.figure(figsize=(8, 6))
        plt.plot(decoder_counts, cov_euc_list, color='blue', label='Euclidean distance')  # Blue line, no marker
        plt.plot(decoder_counts, cov_geo_list, color='orange', label='Geodesic distance')  # Orange line, no marker
        plt.xlabel("Number of decoders")
        plt.ylabel("Coefficient of Variation")
        plt.ylim(0.05, 0.16)  # Match the Y-axis range from the figure
        plt.legend(loc='lower left')  # Position legend in bottom left
        plt.tight_layout()
        plt.savefig("cov_plot_ensemble_size.png")
        plt.show()