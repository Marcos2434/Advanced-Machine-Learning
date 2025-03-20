# Code for DTU course 02460 (Advanced Machine Learning Spring) by Jes Frellsen, 2024
# Version 1.2 (2024-02-06)
# Inspiration is taken from:
# - https://github.com/jmtomczak/intro_dgm/blob/main/vaes/vae_example.ipynb
# - https://github.com/kampta/pytorch-distributions/blob/master/gaussian_vae.py

import torch
import torch.nn as nn
import torch.distributions as td
import torch.utils.data
from torch.nn import functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde

# Gaussian
# python3 vae_bernoulli.py train --device mps --latent-dim 30 --epochs 30 --batch-size 128 --model gaussian_prior.pt --prior g
# python3 vae_bernoulli.py sample --device mps --latent-dim 30 --model gaussian_prior.pt --samples samples_gaussian.png --prior g
# python3 vae_bernoulli.py evaluate_elbo --device mps --latent-dim 30 --model gaussian_prior.pt --prior g
# python3 vae_bernoulli.py evaluate_elbo_mean_std --device mps --latent-dim 30 --epochs 30 --batch-size 128 --prior g

# Mixture of Gaussians
# python3 vae_bernoulli.py train --device mps --latent-dim 30 --epochs 30 --batch-size 128 --model mog_prior.pt --prior m
# python3 vae_bernoulli.py sample --device mps --latent-dim 30 --model mog_prior.pt --samples samples_mog.png --prior m
# python3 vae_bernoulli.py evaluate_elbo --device mps --latent-dim 30 --model mog_prior.pt --prior m
# python3 vae_bernoulli.py evaluate_elbo_mean_std --device mps --latent-dim 30 --epochs 30 --batch-size 128 --prior m

# VampPrior
# python3 vae_bernoulli.py train --device mps --latent-dim 30 --epochs 30 --batch-size 128 --model vamp_prior.pt --prior v
# python3 vae_bernoulli.py sample --device mps --latent-dim 30 --model vamp_prior.pt --samples samples_vamp.png --prior v
# python3 vae_bernoulli.py evaluate_elbo --device mps --latent-dim 30 --model vamp_prior.pt --prior v
# python3 vae_bernoulli.py evaluate_elbo_mean_std --device mps --latent-dim 30 --epochs 30 --batch-size 128 --prior v



import torch
import numpy as np

def train_and_evaluate_multiple_runs(filename, model, optimizer, train_loader, test_loader, epochs, device, num_runs=5):
    """
    Train and evaluate the model multiple times and compute mean and std of ELBO.
    
    Parameters:
    - model: VAE model
    - optimizer: Optimizer for training
    - train_loader: Training data loader
    - test_loader: Test data loader
    - epochs: Number of epochs for training
    - device: Device (cpu/mps/cuda)
    - num_runs: Number of times to repeat training and evaluation
    
    Returns:
    - Mean and standard deviation of ELBO across runs
    """
    elbo_values = []

    for run in range(num_runs):
        print(f"\n=== Training Run {run+1}/{num_runs} ===")
        
        # Reinitialize the model and optimizer for each run
        model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None) # reset parameters instead of instantiating a new model
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        train(model, optimizer, train_loader, epochs, device)

        # evaluate and store ELBO
        test_elbo = model.evaluate_elbo(test_loader, device)
        print(f"Run {run+1}: Test ELBO = {test_elbo:.4f}")
        elbo_values.append(test_elbo)

    elbo_mean = np.mean(elbo_values)
    elbo_std = np.std(elbo_values)

    print(f"\nFinal Report: Test ELBO (Mean ± Std) = {elbo_mean:.4f} ± {elbo_std:.4f}")

    with open(filename, 'w') as f:
        f.write(f"ELBO mean: {elbo_mean:.4f}\n")
        f.write(f"ELBO std: {elbo_std:.4f}\n")
    
    return elbo_mean, elbo_std

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

# Deep Generative Models p.119
# class VampPrior(nn.Module):
#     def __init__(self, L, D, num_vals, encoder, num_components, data=None):
#         super(VampPrior, self).__init__()
        
#         self.L = L  # Latent space dimension
#         self.D = D  # Data dimension
#         self.num_vals = num_vals  # Number of values per dimension, defined the range of the pseudo-inputs
#         self.encoder = encoder  # Encoder model
#         self.num_components = num_components  # Number of mixture components, i.e. pseudo-inputs

#         # Pseudo-inputs, randomly initialized
#         u = torch.rand(num_components, D) * self.num_vals
#         self.u = nn.Parameter(u)  # Trainable pseudo-inputs

#         # Mixing weights for the components
#         self.w = nn.Parameter(torch.zeros(num_components, 1, 1))  # (K x 1 x 1)

#     def get_params(self):
#         """
#         Computes the mean and log-variance of the VampPrior distribution.
#         """
#         mean_vampprior, logvar_vampprior = self.encoder.encode(self.u)  # (K x L), (K x L)
#         return mean_vampprior, logvar_vampprior

#     def sample(self, batch_size):
#         """
#         Samples latent variables from the VampPrior.
#         """
#         mean_vampprior, logvar_vampprior = self.get_params()

#         # Compute mixture probabilities
#         w = F.softmax(self.w, dim=0)  # (K x 1 x 1)
#         w = w.squeeze()  # Remove unnecessary dimensions

#         # Sample mixture components
#         indexes = torch.multinomial(w, batch_size, replacement=True)

#         # Sample latent variables
#         eps = torch.randn(batch_size, self.L)  # Sample standard normal noise
#         for i in range(batch_size):
#             indx = indexes[i]
#             if i == 0:
#                 z = mean_vampprior[[indx]] + eps[[i]] * torch.exp(logvar_vampprior[[indx]])
#             else:
#                 z = torch.cat((z, mean_vampprior[[indx]] + eps[[i]] * torch.exp(logvar_vampprior[[indx]])), 0)
        
#         return z

#     def log_prob(self, z):
#         """
#         Computes the log probability of a given latent variable under the VampPrior.
#         """
#         mean_vampprior, logvar_vampprior = self.get_params()  # (K x L), (K x L)

#         # Compute mixture probabilities
#         w = F.softmax(self.w, dim=0)  # (K x 1 x 1)

#         # Expand dimensions to match batch size
#         z = z.unsqueeze(0)  # (1 x B x L)
#         mean_vampprior = mean_vampprior.unsqueeze(1)  # (K x 1 x L)
#         logvar_vampprior = logvar_vampprior.unsqueeze(1)  # (K x 1 x L)

#         # Compute log probabilities
#         log_p = -0.5 * (torch.sum((z - mean_vampprior) ** 2 / torch.exp(logvar_vampprior), dim=-1) + torch.sum(logvar_vampprior, dim=-1))
#         log_p += torch.log(w).squeeze()  # (K x B)
#         log_prob = torch.logsumexp(log_p, dim=0, keepdim=False)  # (B)
        
#         return log_prob

# Deep Generative Models p.119 with modified sample, log_prob and forward methods
class VampPrior(nn.Module):
    def __init__(self, L, D, num_vals, encoder, num_components, data=None):
        super(VampPrior, self).__init__()
        
        self.L = L  # Latent space dimension
        self.D = D  # Data dimension
        self.num_vals = num_vals  # Number of values per dimension, defines the range of the pseudo-inputs
        self.encoder = encoder  # Encoder model
        self.num_components = num_components  # Number of mixture components, i.e., pseudo-inputs

        # Pseudo-inputs, randomly initialized
        u = torch.rand(num_components, D) * self.num_vals
        self.u = nn.Parameter(u)  # Trainable pseudo-inputs
    
        # Mixing weights for the components
        self.w = nn.Parameter(torch.zeros(num_components, 1, 1))  # (K x 1 x 1)

    def get_params(self):
        posterior_dist = self.encoder(self.u.unsqueeze(-1)) # pass pseudo-inputs through the encoder
        mean_vampprior, logvar_vampprior = posterior_dist.mean, posterior_dist.stddev # obtain mean and log-variance
        return mean_vampprior, logvar_vampprior

    def forward(self):
        """
        Return the prior distribution as a mixture of Gaussians.
        """
        mean_vampprior, logvar_vampprior = self.get_params()

        # compute mixture probabilities
        w = F.softmax(self.w, dim=0)  # (K x 1 x 1)

        # create a mixture of Gaussians
        mixture_dist = td.Categorical(probs=w.squeeze())  # (K)
        component_dist = td.Independent(td.Normal(loc=mean_vampprior, scale=torch.exp(0.5 * logvar_vampprior)), 1)  # (K x L)

        return td.MixtureSameFamily(mixture_dist, component_dist)

    def sample(self, batch_size):
        """
        Samples latent variables from the VampPrior.
        """
        prior = self.forward()
        return prior.sample(torch.Size([batch_size]))

    def log_prob(self, z):
        """
        Computes the log probability of a given latent variable under the VampPrior.
        """
        prior = self.forward()
        return prior.log_prob(z)

class MixtureOfGaussiansPrior(nn.Module):
    def __init__(self, M, num_components=5):
        """
        Define a Mixture of Gaussians (MoG) prior.

        Parameters:
        M: [int] 
           Dimension of the latent space.
        num_components: [int] 
           Number of Gaussian components in the mixture.
        """
        super(MixtureOfGaussiansPrior, self).__init__()
        self.M = M
        self.num_components = num_components
        
        # mixture component means and stds (learnable)
        self.means = nn.Parameter(torch.randn(num_components, M), requires_grad=True)
        self.stds = nn.Parameter(torch.ones(num_components, M), requires_grad=True)
        
        # mixture weights (softmax applied to ensure they sum to 1)
        self.mixture_weights = nn.Parameter(torch.ones(num_components) / num_components)  # equal weights initially

    def forward(self):
        """
        Return the MoG prior distribution.
        """
        # Create a mixture distribution
        mixture_dist = td.Categorical(logits=self.mixture_weights) # each gaussian has equal weights initially
        component_dist = td.Independent(td.Normal(loc=self.means, scale=torch.exp(self.stds)), 1) # the distributions themselves
        return td.MixtureSameFamily(mixture_dist, component_dist)

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
        
        self.min_std = 1e-3

    def forward(self, x):
        """
        Given a batch of data, return a Gaussian distribution over the latent space.

        Parameters:
        x: [torch.Tensor] 
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        mean, std = torch.chunk(self.encoder_net(x), 2, dim=-1) # split the output into mean and std
        std = F.softplus(std) + self.min_std
        return td.Independent(td.Normal(loc=mean, scale=torch.exp(std)), 1) # return a Gaussian distribution over the latent space


class BernoulliDecoder(nn.Module):
    def __init__(self, decoder_net):
        """
        Define a Bernoulli decoder distribution based on a given decoder network.

        Parameters: 
        encoder_net: [torch.nn.Module]             
           The decoder network that takes as a tensor of dim `(batch_size, M) as
           input, where M is the dimension of the latent space, and outputs a
           tensor of dimension (batch_size, feature_dim1, feature_dim2).
        """
        super(BernoulliDecoder, self).__init__()
        self.decoder_net = decoder_net
        self.std = nn.Parameter(torch.ones(28, 28)*0.5, requires_grad=True) # 28x28 is the size of the image, 0.5 is the variance

    def forward(self, z):
        """
        Given a batch of latent variables, return a Bernoulli distribution over the data space.

        Parameters:
        z: [torch.Tensor] 
           A tensor of dimension `(batch_size, M)`, where M is the dimension of the latent space.
        """
        logits = self.decoder_net(z)
        return td.Independent(td.Bernoulli(logits=logits), 2)


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
        q = self.encoder(x) # encoded posterior q(z|x)
        z = q.rsample() # sample z from posterior
        prior = self.prior() # prior p(z)
        
        log_px_z = self.decoder(z).log_prob(x) # reconstruction term (log p(x|z)) (log likelihood)

        if isinstance(prior, td.MixtureSameFamily) or isinstance(prior, VampPrior):
            log_qz = q.log_prob(z) # log q(z|x)
            log_pz = prior.log_prob(z) # log p(z)

            kl_div = (log_qz - log_pz).mean() # MC approximation of KL divergence
        else:
            kl_div = td.kl_divergence(q, prior).mean()

        elbo = log_px_z.mean() - kl_div
        return elbo
    
    def evaluate_elbo(self, data_loader, device):
        """
        Computes ELBO for each batch in the test set and averages over all batches to estimate the model's overall ELBO.
        Higher ELBO → The model is both reconstructing well and learning a structured latent space.
        Lower ELBO → Either poor reconstructions (low log-likelihood) or bad latent space organization (high KL divergence).
        """
        self.eval()
        total_elbo = 0
        num_batches = 0
        with torch.no_grad():
            for x, _ in data_loader:
                x = x.to(device)
                total_elbo += self.elbo(x).item()
                num_batches += 1
        return total_elbo / num_batches

    

    # def plot_distribution(self, distribution, data_loader, device, latent_dim):
    #     from scipy.stats import gaussian_kde
    #     self.eval()
    #     all_z = []
    #     all_labels = []
    #     cmap = None
    #     with torch.no_grad():
    #         if distribution == 'prior':
    #             for x, labels in data_loader:
    #                 x = x.to(device)
    #                 z_samples = self.prior().sample(torch.Size([x.size(0)]))
    #                 all_z.append(z_samples.cpu().numpy())
    #                 all_labels.append(labels.numpy())
    #             c = 'black'
    #             cmap = None  # No colormap for prior
    #         elif distribution == 'posterior':
    #             for x, labels in data_loader:
    #                 x = x.to(device)
    #                 q = self.encoder(x)
    #                 z_samples = q.rsample().cpu().numpy()
    #                 all_z.append(z_samples)
    #                 all_labels.append(labels.numpy())
    #             c = np.concatenate(all_labels, axis=0)
    #             cmap = 'tab10'

    #     all_z = np.concatenate(all_z, axis=0)
        
    #     # If latent dim > 2, reduce dimensions with PCA
    #     if latent_dim > 2:
    #         pca = PCA(n_components=2)
    #         all_z = pca.fit_transform(all_z)
        
    #     x_vals, y_vals = all_z[:, 0], all_z[:, 1]
        
    #     # Compute density using KDE
    #     kde = gaussian_kde(np.vstack([x_vals, y_vals]))
    #     x_grid, y_grid = np.meshgrid(
    #         np.linspace(x_vals.min(), x_vals.max(), 100),
    #         np.linspace(y_vals.min(), y_vals.max(), 100)
    #     )
    #     z_grid = kde(np.vstack([x_grid.ravel(), y_grid.ravel()])).reshape(x_grid.shape)
        
    #     plt.figure(figsize=(8, 6))
    #     scatter = plt.scatter(x_vals, y_vals, c=c, alpha=0.6, cmap=cmap)
    #     plt.colorbar(scatter, label="Digit Label")
        
    #     # Add contour plot
    #     plt.contour(x_grid, y_grid, z_grid, levels=10, cmap='coolwarm', alpha=0.7)
        
    #     plt.xlabel("Latent Dimension 1")
    #     plt.ylabel("Latent Dimension 2")
    #     plt.title("Latent Space Projection with Contours")
    #     plt.show()
    
    def plot_prior_and_aggregate_posterior(self, test_loader, device, latent_dim):
        """
        Plot the prior distribution using contour plots and overlay samples from the aggregate posterior.
        """
        self.eval()
        with torch.no_grad():
            # Sample from prior
            prior = self.prior()
            prior_samples = prior.sample((1000,)).cpu().numpy()

            # Sample from aggregate posterior
            posterior_samples = []
            for x, _ in test_loader:
                x = x.to(device)
                q_z_x = self.encoder(x)
                z_samples = q_z_x.rsample().cpu().numpy()
                posterior_samples.append(z_samples)
            posterior_samples = np.concatenate(posterior_samples, axis=0)

        # Apply PCA if latent_dim > 2
        if latent_dim > 2:
            pca = PCA(n_components=2)
            prior_samples = pca.fit_transform(prior_samples)
            posterior_samples = pca.transform(posterior_samples)

        # Plot contour for prior distribution
        plt.figure(figsize=(8, 6))
        plt.scatter(prior_samples[:, 0], prior_samples[:, 1], alpha=0.3, label="Prior Samples")
        plt.scatter(posterior_samples[:, 0], posterior_samples[:, 1], alpha=0.5, color="red", label="Posterior Samples")
        plt.xlabel("Latent Dim 1")
        plt.ylabel("Latent Dim 2")
        plt.title("Prior Distribution with Aggregate Posterior Samples")
        plt.legend()
        plt.show()

    def plot_aggregate_posterior_with_labels(self, test_loader, device, latent_dim):
        """
        Plot samples from the aggregate posterior, colored by their true class label.
        """
        self.eval()
        with torch.no_grad():
            posterior_samples = []
            labels = []
            for x, y in test_loader:
                x = x.to(device)
                q_z_x = self.encoder(x)
                z_samples = q_z_x.rsample().cpu().numpy()
                posterior_samples.append(z_samples)
                labels.append(y.numpy())

            posterior_samples = np.concatenate(posterior_samples, axis=0)
            labels = np.concatenate(labels, axis=0)

        # Apply PCA if latent_dim > 2
        if latent_dim > 2:
            pca = PCA(n_components=2)
            posterior_samples = pca.fit_transform(posterior_samples)

        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(posterior_samples[:, 0], posterior_samples[:, 1], c=labels, cmap="tab10", alpha=0.6)
        plt.xlabel("Latent Dim 1")
        plt.ylabel("Latent Dim 2")
        plt.title("Aggregate Posterior Samples Colored by Class")
        plt.colorbar(scatter, label="Class Label")
        plt.show()
        
    def plot_prior_contours(self, test_loader, device, latent_dim):
        """
        Plot the prior distribution using contour plots and overlay samples from the aggregate posterior.
        """
        self.eval()
        with torch.no_grad():
            # Sample from prior
            prior = self.prior()
            prior_samples = prior.sample((1000,)).cpu().numpy()

            # Sample from aggregate posterior
            posterior_samples = []
            for x, _ in test_loader:
                x = x.to(device)
                q_z_x = self.encoder(x)
                z_samples = q_z_x.rsample().cpu().numpy()
                posterior_samples.append(z_samples)
            posterior_samples = np.concatenate(posterior_samples, axis=0)

        # Apply PCA if latent_dim > 2
        if latent_dim > 2:
            pca = PCA(n_components=2)
            prior_samples = pca.fit_transform(prior_samples)
            posterior_samples = pca.transform(posterior_samples)
            
            
        posterior_samples = posterior_samples[:5000]

            
        xvals_posterior, yvals_posterior = posterior_samples[:, 0], posterior_samples[:, 1]
        x_vals, y_vals = prior_samples[:, 0], prior_samples[:, 1]

        # Compute density using KDE
        kde = gaussian_kde(np.vstack([x_vals, y_vals]))
        x_grid, y_grid = np.meshgrid(
            np.linspace(xvals_posterior.min(), xvals_posterior.max(), 100),
            np.linspace(yvals_posterior.min(), yvals_posterior.max(), 100)
        )
        z_grid = kde(np.vstack([x_grid.ravel(), y_grid.ravel()])).reshape(x_grid.shape)

        plt.figure(figsize=(8, 6))
        plt.contourf(x_grid, y_grid, z_grid, levels=100, cmap='viridis')
        plt.scatter(posterior_samples[:, 0], posterior_samples[:, 1], alpha=0.6, color='black', label='Posterior Samples', s=3)
        # range of x and y between -2 and 2

        plt.xlabel("Latent Dim 1")
        plt.ylabel("Latent Dim 2")
        plt.title("Prior Distribution Contours with Aggregate Posterior Samples")
        plt.colorbar(label="Density")
        plt.legend()
        plt.show()


    
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
    model.train()

    total_steps = len(data_loader)*epochs
    progress_bar = tqdm(range(total_steps), desc="Training")

    for epoch in range(epochs):
        data_iter = iter(data_loader)
        for x in data_iter:
            x = x[0].to(device)
            optimizer.zero_grad()
            loss = model(x)
            loss.backward()
            optimizer.step()

            # Update progress bar
            progress_bar.set_postfix(loss=f"⠀{loss.item():12.4f}", epoch=f"{epoch+1}/{epochs}")
            progress_bar.update()


if __name__ == "__main__":
    from torchvision import datasets, transforms
    from torchvision.utils import save_image, make_grid
    import glob

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, default='train', choices=['train', 'sample', 'evaluate_elbo', 'evaluate_elbo_mean_std'], help='what to do when running the script (default: %(default)s)')
    parser.add_argument('--model', type=str, default='model.pt', help='file to save model to or load model from (default: %(default)s)')
    parser.add_argument('--samples', type=str, default='samples.png', help='file to save samples in (default: %(default)s)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], help='torch device (default: %(default)s)')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N', help='batch size for training (default: %(default)s)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: %(default)s)')
    parser.add_argument('--latent-dim', type=int, default=32, metavar='N', help='dimension of latent variable (default: %(default)s)')
    parser.add_argument('--prior', type=str, default='g', choices=['g', 'm', 'v'], help='prior to use (default: %(default)s)')

    args = parser.parse_args()
    print('# Options')
    for key, value in sorted(vars(args).items()):
        print(key, '=', value)

    device = args.device

    # Load MNIST as binarized at 'thresshold' and create data loaders
    thresshold = 0.5
    mnist_train_loader = torch.utils.data.DataLoader(datasets.MNIST('data/', train=True, download=True,
                                                                    transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: (thresshold < x).float().squeeze())])),
                                                    batch_size=args.batch_size, shuffle=True)
    mnist_test_loader = torch.utils.data.DataLoader(datasets.MNIST('data/', train=False, download=True,
                                                                transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: (thresshold < x).float().squeeze())])),
                                                    batch_size=args.batch_size, shuffle=True)

    # Define prior distribution
    M = args.latent_dim

    # Define encoder and decoder networks
    encoder_net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, M*2),  # M*2 because we need to split the output into mean and std 
    )

    decoder_net = nn.Sequential(
        nn.Linear(M, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 784),
        nn.Unflatten(-1, (28, 28))
    )

    # Define VAE model
    # the nets output the parameters of the distributions for the bernoulli and gaussian distributions
    decoder = BernoulliDecoder(decoder_net)
    encoder = GaussianEncoder(encoder_net)
    
    # CHOOSE PRIOR
    if args.prior == 'g':  prior = GaussianPrior(M)
    elif args.prior == 'm': prior = MixtureOfGaussiansPrior(M)
    elif args.prior == 'v': prior = VampPrior(L=M, D=784, num_vals=20, encoder=encoder, num_components=1000)

    model = VAE(prior, decoder, encoder).to(device)
        
    # Choose mode to run
    if args.mode == 'train':
        # Define optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Train model
        train(model, optimizer, mnist_train_loader, args.epochs, args.device)

        # Save model
        torch.save(model.state_dict(), args.model)

    elif args.mode == 'sample':
        model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))

        # Generate samples
        model.eval()
        with torch.no_grad():
            samples = (model.sample(4)).cpu() 
            save_image(samples.view(4, 1, 28, 28), args.samples)
    
    elif args.mode == 'evaluate_elbo':
        # python3 vae_bernoulli.py evaluate_elbo --device cpu --latent-dim 10 --model model.pt
        model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))
        test_elbo = model.evaluate_elbo(mnist_test_loader, device)
        print(f"Test ELBO: {test_elbo:.4f}")
        
        model.plot_prior_contours(mnist_test_loader, device, latent_dim=M)
        # model.plot_prior_and_aggregate_posterior(mnist_test_loader, device, M)
        # model.plot_aggregate_posterior_with_labels(mnist_test_loader, device, M)
    
    elif args.mode == 'evaluate_elbo_mean_std':
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        train_and_evaluate_multiple_runs("gaussian_prior_elbo", model, optimizer, mnist_train_loader, mnist_test_loader, args.epochs, device, num_runs=10)