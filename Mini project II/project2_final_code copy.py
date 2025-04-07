import torch
import torch.nn as nn
import torch.distributions as td
import torch.utils.data
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import numpy as np
from geodesics_utilsbis import optimize_geodesic, compute_energy, model_average_energy, compute_length, plot_geodesics
import random

"""
    How to run:
    python project2_final_code.py geodesics_single --device cuda --num-curves 25 --num-t 128 --num-iters 50 --lr 5e-2 --num-decoders 3
    python project2_final_code.py geodesics_ensemble --device cuda --num-curves 25 --num-t 128 --num-iters 50 --lr 5e-2 --num-decoders 3

    python project2_final_code.py cov_plot --device cuda --num-t 15 --num-iters 50 --lr 1e-2
"""

""" mps
    How to run:
    python project2_final_code.py geodesics_single --device mps --num-curves 2 --num-t 15 --num-iters 50 --lr 1e-2 --num-decoders 10
    python project2_final_code.py geodesics_ensemble --device mps --num-curves 2 --num-t 15 --num-iters 50 --lr 1e-2 --num-decoders 10

    python project2_final_code.py cov_plot --device mps --num-t 15 --num-iters 50 --lr 1e-2
"""
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

    def elbo(self, x, decoder_idx):
        q = self.encoder(x)  
        z = q.rsample()  
        decoder = self.decoders[decoder_idx]
        elbo = torch.mean(decoder(z).log_prob(x) - q.log_prob(z) + self.prior().log_prob(z))  # Compute ELBO using the likelihood, posterior and prior
        return elbo

    def sample(self, n_samples=1, decoder_idx=None):
        z = self.prior().sample(torch.Size([n_samples]))
        if decoder_idx is None:
            decoder_idx = torch.randint(0, self.num_decoders, (1,)).item()
        return self.decoders[decoder_idx](z).sample()

    def forward(self, x, decoder_idx):
        return -self.elbo(x, decoder_idx)

def train(model, optimizers, data_loader, epochs_per_decoder, device):
    total_epochs = epochs_per_decoder * model.num_decoders
    num_steps = len(data_loader) * total_epochs
    epoch = 0

    def noise(x, std=0.05):
        eps = std * torch.randn_like(x)
        return torch.clamp(x + eps, min=0.0, max=1.0)

    with tqdm(range(num_steps)) as pbar:
        for step in pbar:
            try:
                x = next(iter(data_loader))[0]
                x = noise(x.to(device))
                decoder_idx = torch.randint(0, model.num_decoders, (1,)).item()
                optimizer = optimizers[decoder_idx]
                optimizer.zero_grad()
                loss = model(x, decoder_idx)
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
        choices=["train", "sample", "eval", "geodesics_single", "geodesics_ensemble", "cov_plot"],
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
        optimizers = [
            torch.optim.Adam(list(model.encoder.parameters()) + list(decoder.parameters()), lr=args.lr) for decoder in model.decoders
        ]

        print(f"Training VAE with {args.num_decoders} decoders")
        train(model, optimizers, mnist_train_loader, args.epochs_per_decoder, device)
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
            optimizer = [torch.optim.Adam(single_decoder_model.parameters(), lr=args.lr)]
            train(single_decoder_model, optimizer, mnist_train_loader, args.epochs_per_decoder, device)
            torch.save(single_decoder_model.state_dict(), single_decoder_path)

    if args.mode == "geodesics_single":
        # Load the single-decoder VAE
        single_decoder_path = f"{args.experiment_folder}/vae_single_decoder.pt"
        single_decoder_model = VAE(
            GaussianPrior(M),
            [GaussianDecoder(new_decoder())],
            GaussianEncoder(new_encoder())
        ).to(device)
        single_decoder_model.load_state_dict(torch.load(single_decoder_path))
        single_decoder_model.eval()

        # Step 0: Load and project all test points into latent space
        test_latents = []
        test_labels = []
        data_iter = iter(mnist_test_loader)
        for x_batch, labels in mnist_test_loader:
            x_batch = x_batch.to(device)
            with torch.no_grad():
                # Use the encoder from the single-decoder model (same encoder for both modes)
                z = single_decoder_model.encoder(x_batch).mean  # Shape: (batch_size, 2)
            test_latents.append(z)
            test_labels.append(labels)
        test_latents = torch.cat(test_latents, dim=0).cpu()  # Shape: (2048, 2)
        test_labels = torch.cat(test_labels, dim=0).cpu()  # Shape: (2048,)
        torch.save(test_latents, os.path.join(args.experiment_folder, "test_latents.pt"))
        torch.save(test_labels, os.path.join(args.experiment_folder, "test_labels.pt"))
        
        # Step 1: Generate latent pairs and store their class labels
        latent_pairs = []
        latent_pair_labels = []  # Store (label1, label2) for each pair
        data_iter = iter(mnist_test_loader)
        for _ in range(args.num_curves):
            x_batch, label_batch = next(data_iter)
            x_batch = x_batch.to(device)
            label_batch = label_batch.to(device)
            x1 = x_batch[0].unsqueeze(0)  # Shape: (1, 1, 28, 28)
            x2 = x_batch[1].unsqueeze(0)
            label1 = label_batch[0]  # Class label of x1
            label2 = label_batch[1]  # Class label of x2
            with torch.no_grad():
                z1 = single_decoder_model.encoder(x1).mean.squeeze(0)  # Shape: (2,)
                z2 = single_decoder_model.encoder(x2).mean.squeeze(0)
            latent_pairs.append((z1, z2))
            latent_pair_labels.append((label1, label2))
        torch.save(latent_pairs, os.path.join(args.experiment_folder, "latent_pairs_single.pt"))
        torch.save(latent_pair_labels, os.path.join(args.experiment_folder, "latent_pair_labels.pt"))
        print(f"Saved latent points in {args.experiment_folder}/latent_pairs_single.pt")

        latent_pairs = torch.load(os.path.join(args.experiment_folder, "latent_pairs_single.pt"))
        latent_pair_labels = torch.load(os.path.join(args.experiment_folder, "latent_pair_labels.pt"))

        # Step 2: Compute geodesics for single decoder
        single_decoder = single_decoder_model.decoders[0]
        single_geodesics = []
        geodesics_dir = f'{args.experiment_folder}/geodesics_single_num-t={args.num_t}'
        os.makedirs(geodesics_dir, exist_ok=True)
        for curve_idx, (z1, z2) in enumerate(latent_pairs[:args.num_curves]):
            print(f"Computing geodesic #{curve_idx+1}/{len(latent_pairs)} (Single Decoder)")
            geodesic_path = f'{geodesics_dir}/geodesic_single_{curve_idx+1}.pt'
            if os.path.exists(geodesic_path):
                geodesic = torch.load(geodesic_path, map_location=device)
                print(f"Geodesic loaded from {geodesic_path}")
            else:
                geodesic = optimize_geodesic(
                    z_start=z1,
                    z_end=z2,
                    num_points=args.num_t,
                    num_iters=args.num_iters,
                    lr=args.lr,
                    energy_fn=lambda c: compute_energy(single_decoder, c),
                    convergence_threshold=1e-3,
                    window_size=10
                )
                torch.save(geodesic, geodesic_path)
            single_geodesics.append(geodesic)

        # Step 3: Plot using the consolidated function
        plot_geodesics(
            latent_pairs=latent_pairs,
            latent_pair_labels=latent_pair_labels,
            geodesics=single_geodesics,
            test_latents=test_latents,
            test_labels=test_labels,
            model=single_decoder_model,
            mode=args.mode,
            num_decoders=1,
            device=device,
            experiment_folder=args.experiment_folder,
            filename_suffix=f"num-t={args.num_t}"
        )
        

    elif args.mode == "geodesics_ensemble":
        # Load the ensemble VAE
        ensemble_path = f"{args.experiment_folder}/vae_with_{args.num_decoders}_decoders.pt"
        ensemble_model = VAE(
            GaussianPrior(M),
            [GaussianDecoder(new_decoder()) for _ in range(args.num_decoders)],
            GaussianEncoder(new_encoder())
        ).to(device)
        ensemble_model.load_state_dict(torch.load(ensemble_path))
        ensemble_model.eval()

        # Step 1: Generate 25 random latent pairs
        # latent_pairs = []
        # data_iter = iter(mnist_test_loader)
        # for _ in range(args.num_curves):
        #     x_batch, _ = next(data_iter)
        #     x_batch = x_batch.to(device)
        #     x1 = x_batch[0].unsqueeze(0)  # Shape: (1, 1, 28, 28)
        #     x2 = x_batch[1].unsqueeze(0)
        #     with torch.no_grad():
        #         z1 = ensemble_model.encoder(x1).mean.squeeze(0)  # Shape: (2,)
        #         z2 = ensemble_model.encoder(x2).mean.squeeze(0)
        #     latent_pairs.append((z1, z2))
        # torch.save(latent_pairs, os.path.join(args.experiment_folder, "latent_pairs_ensemble.pt"))

        
        # latent_pairs = []
        # latent_pair_labels = []
        # z1 = torch.tensor([-2.5, 1.0], device=device)
        # z2 = torch.tensor([0.0, 5.0], device=device)
        # label1 = torch.tensor(1, device=device)
        # label2 = torch.tensor(0, device=device)
        # latent_pairs.append((z1, z2))
        # latent_pair_labels.append((label1, label2))

        # # Save the latent pairs with labels
        # torch.save(latent_pairs, os.path.join(args.experiment_folder, "latent_pairs_with_labels_ensemble.pt"))
        # print(f"Saved {len(latent_pairs)} latent pairs with labels to latent_pairs_with_labels_ensemble.pt")
        
        latent_pairs = torch.load(os.path.join(args.experiment_folder, "latent_pairs_single.pt"))
        latent_pair_labels = torch.load(os.path.join(args.experiment_folder, "latent_pair_labels.pt"))

        # Step 2: Compute geodesics for ensemble decoder
        ensemble_decoders = ensemble_model.decoders
        ensemble_geodesics = []
        geodesics_dir = f'{args.experiment_folder}/geodesics_ensemble_num-t={args.num_t}'
        os.makedirs(geodesics_dir, exist_ok=True)
        for curve_idx, (z1, z2) in enumerate(latent_pairs):
            print(f"Computing geodesic #{curve_idx+1}/{len(latent_pairs)} (Ensemble Decoder)")
            geodesic = optimize_geodesic(
                z_start=z1,
                z_end=z2,
                num_points=args.num_t,
                num_iters=args.num_iters,
                lr=args.lr,
                energy_fn=lambda c: model_average_energy(c, ensemble_decoders, num_samples=1),
                convergence_threshold=1e-3,
                window_size=10
            )
            ensemble_geodesics.append(geodesic)
            geodesic_path = f'{geodesics_dir}/geodesic_ensemble_{curve_idx+1}.pt'
            torch.save(geodesic, geodesic_path)

        # ensemble_geodesics = []
        # for i in range(25):
        #     geodesic_path = f'{geodesics_dir}/geodesic_ensemble_{i+1}.pt'
        #     geodesic = torch.load(geodesic_path)  
        #     ensemble_geodesics.append(geodesic)


        test_latents = torch.load(os.path.join(args.experiment_folder, "test_latents.pt"))
        test_labels = torch.load(os.path.join(args.experiment_folder, "test_labels.pt"))

        # Step 3: Plotting for Ensemble Decoder with Std Background
        plot_geodesics(
            latent_pairs=latent_pairs,
            latent_pair_labels=latent_pair_labels,
            geodesics=ensemble_geodesics,
            test_latents=test_latents,
            test_labels=test_labels,
            model=ensemble_model,
            mode='geodesics_ensemble',
            num_decoders=args.num_decoders,
            device=device,
            experiment_folder=args.experiment_folder,
            filename_suffix=f"num-t={args.num_t}"
        )

    elif args.mode == "cov_plot":
        # Define ensemble sizes to evaluate
        max_decoders = 3  # Maximum number of decoders
        decoder_counts = list(range(1, max_decoders + 1))  # [1, 2, ..., 3]
        num_vae = 10  # Number of VAEs
        num_pairs = 10  # Number of test point pairs

        cov_geo_list = []  # Geodesic CoV for each ensemble size
        cov_euc_list = []  # Euclidean CoV for each ensemble size

        # Step 1: Train M=10 VAEs with max_decoders (3) decoders if not already trained
        experiments_folder = f"{args.experiment_folder}/ensemble_max_decoders"
        os.makedirs(experiments_folder, exist_ok=True)

        # Train M VAEs with max_decoders
        for m in range(num_vae):
            model_path = f"{experiments_folder}/vae_{m}.pt"
            if not os.path.exists(model_path):
                print(f"Training VAE #{m+1}/{num_vae} with {max_decoders} decoders")
                decoders = [GaussianDecoder(new_decoder()) for _ in range(max_decoders)]
                model = VAE(GaussianPrior(M), decoders, GaussianEncoder(new_encoder())).to(device)
                optimizers = [
                    torch.optim.Adam(list(model.encoder.parameters()) + list(decoder.parameters()), lr=args.lr) for decoder in model.decoders
                ]
                train(model, optimizers, mnist_train_loader, args.epochs_per_decoder, device)
                torch.save(model.state_dict(), model_path)

        # Step 2: Load the M VAEs and store their encoders and decoders
        models = []
        for m in range(num_vae):
            model = VAE(
                GaussianPrior(M),
                [GaussianDecoder(new_decoder()) for _ in range(max_decoders)],
                GaussianEncoder(new_encoder())
            ).to(device)
            model.load_state_dict(torch.load(f"{experiments_folder}/vae_{m}.pt"))
            model.eval()
            models.append(model)

        # Step 3: Select test point pairs (in input space)
        test_dataset = mnist_test_loader.dataset
        dataset_size = len(test_dataset)

        test_pairs = []
        random.seed(42)
        for _ in range(num_pairs):
            # Randomly select two different indices from the entire dataset
            idx1, idx2 = random.sample(range(dataset_size), 2)
            x1, _ = test_dataset[idx1]  # Get sample (tensor, label)
            x2, _ = test_dataset[idx2]
            x1 = x1.unsqueeze(0).to(device)  # Shape: (1, 1, 28, 28)
            x2 = x2.unsqueeze(0).to(device)  # Shape: (1, 1, 28, 28)
            test_pairs.append((x1, x2))
        torch.save(test_pairs, f"{args.experiment_folder}/test_pairs_cov.pt")

        # Step 4: Compute CoV for each number of decoders
        counter = 0
        geodesics_dir = f"{args.experiment_folder}/geodesics_cov"
        os.makedirs(geodesics_dir, exist_ok=True)
        for num_dec in decoder_counts:
            print(f"\n--- Analyzing Ensemble Size {num_dec} Decoders ---")

            geo_lengths_per_pair = []  # List of lists: [pair1_lengths, pair2_lengths, ...]
            euc_dists_per_pair = []    # List of lists: [pair1_dists, pair2_dists, ...]

            # For each pair of test points
            for pair_idx, (x1, x2) in enumerate(test_pairs):
                print(f"Processing pair {pair_idx+1}/{num_pairs}")
                geo_lengths = []  # Geodesic lengths for this pair across M VAEs
                euc_dists = []    # Euclidean distances for this pair across M VAEs

                # Compute distances using each VAE
                for model_idx, model in enumerate(models):
                    # Encode test points with the current model's encoder
                    with torch.no_grad():
                        z1 = model.encoder(x1).mean.squeeze(0)  # Shape: (2,)
                        z2 = model.encoder(x2).mean.squeeze(0)

                    # Euclidean distance (varies across models due to different encoders)
                    euc_dist = torch.norm(z2 - z1).item()
                    euc_dists.append(euc_dist)

                    geodesic_path = f"{geodesics_dir}/geodesic_{counter}.pt"
                    counter += 1
                    if not os.path.exists(geodesic_path):
                        # Geodesic distance (using the first num_dec decoders)
                        decoders = model.decoders[:num_dec]  # Subset of decoders
                        geod = optimize_geodesic(
                            z_start=z1,
                            z_end=z2,
                            num_points=args.num_t,
                            num_iters=args.num_iters,
                            lr=args.lr,
                            energy_fn=lambda c: model_average_energy(c, decoders, num_samples=1),
                            convergence_threshold=1e-3,
                            window_size=10
                        )
                        torch.save(geod, geodesic_path)
                    else:
                        geod = torch.load(geodesic_path)
                        print(f"Loaded geodesic from {geodesic_path}")
                        
                    # Compute average geodesic length over the subset of decoders
                    geo_length = sum(compute_length(dec, geod) for dec in decoders) / len(decoders)
                    geo_lengths.append(geo_length)

                    print(f"  VAE {model_idx+1}: Euclidean={euc_dist:.4f}, Geodesic={geo_length:.4f}")

                # Compute and print CoV for this pair
                geo_lengths_tensor = torch.tensor(geo_lengths)  # Shape: (num_vae,)
                euc_dists_tensor = torch.tensor(euc_dists)      # Shape: (num_vae,)
                geo_cov_pair = (geo_lengths_tensor.std() / geo_lengths_tensor.mean()).item() if geo_lengths_tensor.mean() != 0 else float('nan')
                euc_cov_pair = (euc_dists_tensor.std() / euc_dists_tensor.mean()).item() if euc_dists_tensor.mean() != 0 else float('nan')
                print(f"  Pair {pair_idx+1} CoV: Geodesic={geo_cov_pair:.4f}, Euclidean={euc_cov_pair:.4f}")

                geo_lengths_per_pair.append(geo_lengths)
                euc_dists_per_pair.append(euc_dists)

            # Compute CoV for this ensemble size
            geo_lengths_tensor = torch.tensor(geo_lengths_per_pair)  # Shape: (num_pairs, num_vae)
            euc_dists_tensor = torch.tensor(euc_dists_per_pair)      # Shape: (num_pairs, num_vae)

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

    elif args.mode == 'sample':
        outputs_ensemble = f"{args.experiment_folder}/vae_ensemble_samples"
        os.makedirs(outputs_ensemble, exist_ok=True)

        ensemble_path = f"{args.experiment_folder}/vae_with_{args.num_decoders}_decoders.pt"
        ensemble_model = VAE(
            GaussianPrior(M),
            [GaussianDecoder(new_decoder()) for _ in range(args.num_decoders)],
            GaussianEncoder(new_encoder())
        ).to(device)
        ensemble_model.load_state_dict(torch.load(ensemble_path))
        ensemble_model.eval()

        with torch.no_grad():
            # Sampling from each decoder
            for j in range(ensemble_model.num_decoders):
                samples = ensemble_model.sample(decoder_idx=j, n_samples=64).cpu()
                save_image(samples.view(64, 1, 28, 28), f"{outputs_ensemble}/samples_decoder{j}.png")

            # Reconstruction using each decoder
            data = next(iter(mnist_test_loader))[0].to(device)
            z = ensemble_model.encoder(data).mean
            for j in range(ensemble_model.num_decoders):
                recon = ensemble_model.decoders[j](z).mean
                save_image(
                    torch.cat([data.cpu(), recon.cpu()], dim=0),
                    f"{outputs_ensemble}/reconstruction_decoder{j}.png"
            )
    elif args.mode == "cov_plot_parallel":
        import concurrent.futures
        import numpy as np

        # Define ensemble sizes to evaluate
        max_decoders = 3  # Maximum number of decoders
        decoder_counts = list(range(1, max_decoders + 1))  # [1, 2, 3]
        num_vae = 10  # Number of VAEs
        num_pairs = 10  # Number of test point pairs

        cov_geo_list = []  # Geodesic CoV for each ensemble size
        cov_euc_list = []  # Euclidean CoV for each ensemble size

        experiments_folder = f"{args.experiment_folder}/ensemble_max_decoders"
        os.makedirs(experiments_folder, exist_ok=True)

        # Train M VAEs with max_decoders decoders if not already trained
        for m in range(num_vae):
            model_path = f"{experiments_folder}/vae_{m}.pt"
            if not os.path.exists(model_path):
                print(f"Training VAE #{m+1}/{num_vae} with {max_decoders} decoders")
                decoders = [GaussianDecoder(new_decoder()) for _ in range(max_decoders)]
                model = VAE(GaussianPrior(M), decoders, GaussianEncoder(new_encoder())).to(device)
                optimizers = [
                    torch.optim.Adam(list(model.encoder.parameters()) + list(decoder.parameters()), lr=args.lr)
                    for decoder in model.decoders
                ]
                train(model, optimizers, mnist_train_loader, args.epochs_per_decoder, device)
                torch.save(model.state_dict(), model_path)

        # Load the M VAEs
        models = []
        for m in range(num_vae):
            model = VAE(
                GaussianPrior(M),
                [GaussianDecoder(new_decoder()) for _ in range(max_decoders)],
                GaussianEncoder(new_encoder())
            ).to(device)
            model.load_state_dict(torch.load(f"{experiments_folder}/vae_{m}.pt"))
            model.eval()
            models.append(model)

        # Select test point pairs (in input space)
        test_dataset = mnist_test_loader.dataset
        dataset_size = len(test_dataset)
        test_pairs = []
        random.seed(42)
        for _ in range(num_pairs):
            idx1, idx2 = random.sample(range(dataset_size), 2)
            x1, _ = test_dataset[idx1]
            x2, _ = test_dataset[idx2]
            x1 = x1.unsqueeze(0).to(device)
            x2 = x2.unsqueeze(0).to(device)
            test_pairs.append((x1, x2))
        torch.save(test_pairs, f"{args.experiment_folder}/test_pairs_cov.pt")

        # Directory for storing geodesics
        geodesics_dir = f"{args.experiment_folder}/geodesics_cov"
        os.makedirs(geodesics_dir, exist_ok=True)

        # Define a function to compute geodesic for one VAE on a test pair
        def compute_for_model(pair_idx, model_idx, x1, x2, model, num_dec, args, geodesics_dir):
            with torch.no_grad():
                z1 = model.encoder(x1).mean.squeeze(0)  # Shape: (2,)
                z2 = model.encoder(x2).mean.squeeze(0)
            euc_dist = torch.norm(z2 - z1).item()
            # Use only the first num_dec decoders for the ensemble
            decoders = model.decoders[:num_dec]
            # Create a unique filename for each (ensemble size, pair, model)
            geodesic_path = os.path.join(geodesics_dir, f"geodesic_{num_dec}_{pair_idx}_{model_idx}.pt")
            if not os.path.exists(geodesic_path):
                geod = optimize_geodesic(
                    z_start=z1,
                    z_end=z2,
                    num_points=args.num_t,
                    num_iters=args.num_iters,
                    lr=args.lr,
                    energy_fn=lambda c: model_average_energy(c, decoders, num_samples=1),
                    convergence_threshold=1e-3,
                    window_size=10
                )
                torch.save(geod, geodesic_path)
            else:
                geod = torch.load(geodesic_path)
                print(f"Loaded geodesic from {geodesic_path}")
            geo_length = sum(compute_length(dec, geod) for dec in decoders) / len(decoders)
            return pair_idx, model_idx, euc_dist, geo_length

        # Compute CoV for each ensemble size
        for num_dec in decoder_counts:
            print(f"\n--- Analyzing Ensemble Size {num_dec} Decoders ---")
            # Dictionary to accumulate results per test pair
            results = {}
            # Use ThreadPoolExecutor to parallelize over (pair, model)
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = []
                for pair_idx, (x1, x2) in enumerate(test_pairs):
                    for model_idx, model in enumerate(models):
                        futures.append(executor.submit(
                            compute_for_model, pair_idx, model_idx, x1, x2, model, num_dec, args, geodesics_dir
                        ))
                for future in concurrent.futures.as_completed(futures):
                    pair_idx, model_idx, euc_dist, geo_length = future.result()
                    if pair_idx not in results:
                        results[pair_idx] = {'euc': [], 'geo': []}
                    results[pair_idx]['euc'].append(euc_dist)
                    results[pair_idx]['geo'].append(geo_length)
                    print(f"VAE {model_idx+1} for Pair {pair_idx+1}: Euclidean={euc_dist:.4f}, Geodesic={geo_length:.4f}")

            # Compute CoV per pair then average over pairs
            geo_cov_list_for_num_dec = []
            euc_cov_list_for_num_dec = []
            for pair_idx in sorted(results.keys()):
                euc_arr = np.array(results[pair_idx]['euc'])
                geo_arr = np.array(results[pair_idx]['geo'])
                geo_cov_pair = np.std(geo_arr) / np.mean(geo_arr) if np.mean(geo_arr) != 0 else np.nan
                euc_cov_pair = np.std(euc_arr) / np.mean(euc_arr) if np.mean(euc_arr) != 0 else np.nan
                print(f"Pair {pair_idx+1} CoV: Geodesic={geo_cov_pair:.4f}, Euclidean={euc_cov_pair:.4f}")
                geo_cov_list_for_num_dec.append(geo_cov_pair)
                euc_cov_list_for_num_dec.append(euc_cov_pair)

            geo_cov = np.mean(geo_cov_list_for_num_dec)
            euc_cov = np.mean(euc_cov_list_for_num_dec)
            print(f"Ensemble size {num_dec}: Geodesic CoV={geo_cov:.4f}, Euclidean CoV={euc_cov:.4f}")
            cov_geo_list.append(geo_cov)
            cov_euc_list.append(euc_cov)

        # Step 5: Plot the results
        plt.figure(figsize=(8, 6))
        plt.plot(decoder_counts, cov_euc_list, color='blue', label='Euclidean distance')
        plt.plot(decoder_counts, cov_geo_list, color='orange', label='Geodesic distance')
        plt.xlabel("Number of decoders")
        plt.ylabel("Coefficient of Variation")
        plt.ylim(0.05, 0.16)
        plt.legend(loc='lower left')
        plt.tight_layout()
        plt.savefig("cov_plot_ensemble_size.png")
        plt.show()