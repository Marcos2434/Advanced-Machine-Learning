# Code for DTU course 02460 (Advanced Machine Learning Spring) by Jes Frellsen, 2024
# Version 1.2 (2024-02-06)
# Inspiration is taken from:
# - https://github.com/jmtomczak/intro_dgm/blob/main/vaes/vae_example.ipynb
# - https://github.com/kampta/pytorch-distributions/blob/master/gaussian_vae.py

# Completed by:
# Tischler Ferreira, Martin Lautaro -- s240035

import numpy
import torch
import torch.nn as nn
import torch.distributions as td
import torch.utils.data
from torch.nn import functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np



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

class MoGPrior(nn.Module):
    def __init__(self, latent_dim, num_components):
        """
        Mixture of Gaussian prior p(z) = sum_k w_k N(z; mu_k, sigma_k^2 I).

        Parameters:
        -----------
        latent_dim: int
            Dimension of latent space (M).
        num_components: int
            Number of mixture components (K).
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.num_components = num_components
        
        # Mixture weights (unnormalized logits)
        self.logits = nn.Parameter(torch.zeros(num_components))
        
        # Means for each mixture component: shape (K, M)
        self.means = nn.Parameter(torch.randn(num_components, latent_dim) * 0.01)
        
        # Log-stdevs for each component (so that std = exp(log_stds) > 0)
        self.log_stds = nn.Parameter(torch.zeros(num_components, latent_dim))
        
    def sample(self, n_samples=1):
        """
        Sample from the MoG prior AND return the component indices.
        This is useful for plotting, where we want to color points by component.
        """
        cat = td.Categorical(logits=self.logits)
        component_indices = cat.sample((n_samples,))

        # Select the means and stds corresponding to each sampled component
        means = self.means[component_indices]
        stds = self.log_stds.exp()[component_indices]

        # Sample from N(mean, std)
        z = means + stds * torch.randn_like(means)

        return z, component_indices

    def forward(self):
        """
        Returns a MixtureSameFamily distribution of K Gaussian components in M-dim.
        """
        # Categorical distribution over mixture components
        cat = td.Categorical(logits=self.logits)

        # Wrap the Normals into independent
        component_dist = td.Independent(
            td.Normal(loc=self.means, scale=self.log_stds.exp()), 
            reinterpreted_batch_ndims=1
        )

        # Combine into a Mixture distribution
        return td.MixtureSameFamily(cat, component_dist)

class VampPrior(nn.Module):
    def __init__(self, encoder_net, K=10, input_shape=(1,28,28)):
        """
        VampPrior with K pseudo-inputs, re-using the provided encoder to build
        p(z) = 1/K sum_k=1^K q_phi(z | x^*_k).

        encoder: an nn.Module that maps an image x to (mean, logvar) for q(z|x)
                 i.e. the same as your GaussianEncoder's internal net
        K: number of pseudo inputs
        input_shape: shape of each pseudo input (e.g. (1, 28, 28) for MNIST)
        """
        super().__init__()
        self.encoder_net = encoder_net
        self.K = K
        self.input_shape = input_shape

        # Learnable pseudo-inputs:
        self.pseudo_inputs = nn.Parameter(
            torch.randn(K, *input_shape) * 0.01 # Could change 0.01
        )

    def log_prob(self, z):
        """
        Computes log p(z) = log(1/K sum_k q(z | x^*_k)) using a stable log-sum-exp.
        z should be shape (batch_size, z_dim).
        Returns shape (batch_size,).
        """
        # Flatten pseudo_inputs
        K, *rest = self.pseudo_inputs.shape
        device = z.device

        # Process pseudo-inputs in a single batch:
        x_k = self.pseudo_inputs
        
        # Pass x_k through the encoder network
        encoder_out = self.encoder_net(x_k)
        mean_k, logvar_k = torch.chunk(encoder_out, 2, dim=-1)

        # Compute log-prob
        batch_size, z_dim = z.shape
        z_expand = z.unsqueeze(0)          # shape (1, batch_size, M)
        mean_expand = mean_k.unsqueeze(1)  # shape (K, 1, M)
        logvar_expand = logvar_k.unsqueeze(1)  # shape (K, 1, M)
        log_prob_k = -0.5 * (
            (z_expand - mean_expand)**2 / logvar_expand.exp() 
            + logvar_expand 
            + torch.log(torch.tensor(2.0 * 3.14159265359, device=device))
        ).sum(dim=-1)
        max_ = torch.max(log_prob_k, dim=0, keepdim=True)[0]
        lse = max_ + torch.log(torch.sum(torch.exp(log_prob_k - max_), dim=0, keepdim=True))
        log_p_z = lse.squeeze(0) - torch.log(torch.tensor(K, device=device, dtype=torch.float))

        return log_p_z

    def encoder_net_forward(self, x):
        return self.encoder_net(x)
    
    def sample(self, n_samples=1):
        device = self.pseudo_inputs.device
        with torch.no_grad():
            # Indices for plotting
            k_indices = torch.randint(low=0, high=self.K, size=(n_samples,), device=device)

            x_k = self.pseudo_inputs[k_indices]

            encoder_out = self.encoder_net_forward(x_k)
            mean, logvar = torch.chunk(encoder_out, 2, dim=-1)

            z = mean + torch.randn_like(mean) * torch.exp(0.5 * logvar)

            return z, k_indices


    def forward(self):
        """
        By convention, returning 'self' or an object with a .log_prob() method.  
        This allows your code to do `self.prior().log_prob(z)`.
        """
        return self

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
        self.std = nn.Parameter(torch.ones(28, 28)*0.5, requires_grad=True)

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
        q = self.encoder(x)
        z = q.rsample()
        log_p_x_given_z = self.decoder(z).log_prob(x)

        # Single Gaussian prior
        if isinstance(self.prior, GaussianPrior):
            kl = td.kl_divergence(q, self.prior())
            elbo = log_p_x_given_z - kl

        # Otherwise (MoG or VampPrior) => sample-based
        else:
            if isinstance(self.prior, VampPrior):
                log_p_z = self.prior.log_prob(z)
            else:
                log_p_z = self.prior().log_prob(z)

            log_q_z_given_x = q.log_prob(z)
            elbo = log_p_x_given_z + log_p_z - log_q_z_given_x

        return elbo.mean()

    def sample(self, n_samples=1):
        
        if isinstance(self.prior, VampPrior):
            z, _ = self.prior.sample(n_samples)
        else:
            z = self.prior().sample((n_samples,))
        return self.decoder(z).mean#return z
    
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
            x = x[0].unsqueeze(1).to(device) #For CNN
            optimizer.zero_grad()
            loss = model(x)
            loss.backward()
            optimizer.step()

            # Update progress bar
            progress_bar.set_postfix(loss=f"⠀{loss.item():12.4f}", epoch=f"{epoch+1}/{epochs}")
            progress_bar.update()

def evaluate_elbo(model, data_loader, device):
    model.eval()
    total_elbo = 0
    num_samples = 0

    with torch.no_grad():
        for x, _ in data_loader:
            x = x.unsqueeze(1).to(device)
            elbo = model.elbo(x).item()
            total_elbo += elbo * x.size(0)
            num_samples += x.size(0)

    return total_elbo / num_samples

def plot_latent_space(model, data_loader, device, latent_dim):
    """
    Plot test set samples from the approximate posterior, colored by class label.
    If latent_dim > 2, PCA reduces latent space to 2D.
    """
    model.eval()

    all_z = []
    all_labels = []

    with torch.no_grad():
        for x, y in data_loader:
            x = x.unsqueeze(1).to(device)
            q = model.encoder(x)
            z = q.rsample()
            all_z.append(z.cpu().numpy())
            all_labels.append(y.numpy())

    z = np.concatenate(all_z, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    # Reduce to 2D using PCA
    if latent_dim > 2:
        pca = PCA(n_components=2)
        z = pca.fit_transform(z)

    # Plot
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(z[:, 0], z[:, 1], c=labels, cmap='tab10', s=5)
    plt.colorbar(scatter, label='Class Label')
    plt.xlabel('Latent Dimension 1' if latent_dim <= 2 else 'PCA Component 1')
    plt.ylabel('Latent Dimension 2' if latent_dim <= 2 else 'PCA Component 2')
    plt.title('Latent Space Visualization (Colored by True Class)')
    plt.grid(True)
    plt.show()

def plot_prior_samples(model, device, latent_dim, n_prior_samples=1000):
    model.eval()

    with torch.no_grad():
        if isinstance(model.prior, MoGPrior):
            z_prior, components = model.prior.sample(n_prior_samples)
            components = components.cpu().numpy()

        elif isinstance(model.prior, VampPrior):
            z_prior, components = model.prior.sample(n_prior_samples)
            components = components.cpu().numpy()

        else:
            z_prior = model.prior().sample((n_prior_samples,))
            components = None

    z_prior = z_prior.cpu().numpy()

    if latent_dim > 2:
        pca = PCA(n_components=2)
        z_prior_2d = pca.fit_transform(z_prior)
    else:
        z_prior_2d = z_prior

    plt.figure(figsize=(10, 7))

    if components is not None:
        scatter = plt.scatter(
            z_prior_2d[:, 0], z_prior_2d[:, 1],
            c=components, cmap='tab10', s=5, alpha=0.8
        )
        plt.colorbar(scatter, label="Pseudo-Input (Vamp) / Gaussian Component (MoG)")
    else:
        plt.scatter(z_prior_2d[:, 0], z_prior_2d[:, 1], s=5, alpha=0.8)

    plt.xlabel('Latent Dim 1' if latent_dim <= 2 else 'PCA 1')
    plt.ylabel('Latent Dim 2' if latent_dim <= 2 else 'PCA 2')
    plt.title('Prior Samples (Colored by Pseudo-Input or Component)')
    plt.grid(True)
    plt.show()


########################################
########################################
if __name__ == "__main__":
    from torchvision import datasets, transforms
    from torchvision.utils import save_image, make_grid
    import glob

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, default='train', choices=['train', 'sample','test'], help='what to do when running the script (default: %(default)s)')
    parser.add_argument('--model', type=str, default='model.pt', help='file to save model to or load model from (default: %(default)s)')
    parser.add_argument('--samples', type=str, default='samples.png', help='file to save samples in (default: %(default)s)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], help='torch device (default: %(default)s)')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N', help='batch size for training (default: %(default)s)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: %(default)s)')
    parser.add_argument('--latent-dim', type=int, default=32, metavar='N', help='dimension of latent variable (default: %(default)s)')
    parser.add_argument('--prior-type', type=str, default='gaussian',
                    choices=['gaussian','mog','vamp'], help='Which prior to use.')
    parser.add_argument('--num-mixture-components', type=int, default=1,
        help='Number of mixture components in the prior. If 1, we use a single Gaussian prior.')
    
    args = parser.parse_args()
    print('# Options')
    for key, value in sorted(vars(args).items()):
        print(key, '=', value)

    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: No GPU found. Falling back to CPU.")
        args.device = 'cpu'
    device = args.device

    # Load MNIST as binarized at 'thresshold' and create data loaders
    thresshold = 0.5
    mnist_train_loader = torch.utils.data.DataLoader(datasets.MNIST('data/', train=True, download=True,
                                                                    transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: (thresshold < x).float().squeeze())])),
                                                    batch_size=args.batch_size, shuffle=True)
    mnist_test_loader = torch.utils.data.DataLoader(datasets.MNIST('data/', train=False, download=True,
                                                                transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: (thresshold < x).float().squeeze())])),
                                                    batch_size=args.batch_size, shuffle=True)

    # Get latent dim
    M = args.latent_dim

    # Define encoder and decoder networks
    encoder_net = nn.Sequential(
    nn.Conv2d(1, 32, 4, 2, 1),  # 28x28 → 14x14
    nn.ReLU(),
    nn.Conv2d(32, 64, 4, 2, 1),  # 14x14 → 7x7
    nn.ReLU(),
    nn.Flatten(),
    nn.Linear(64*7*7, 256),
    nn.ReLU(),
    nn.Linear(256, 2*M)
    )

    decoder_net = nn.Sequential(
    nn.Linear(M, 256),
    nn.ReLU(),
    nn.Linear(256, 64*7*7),
    nn.ReLU(),
    nn.Unflatten(-1, (64, 7, 7)),
    nn.ConvTranspose2d(64, 32, 4, 2, 1),  # 7x7 → 14x14
    nn.ReLU(),
    nn.ConvTranspose2d(32, 1, 4, 2, 1),  # 14x14 → 28x28
    )


    # Define VAE model
    decoder = BernoulliDecoder(decoder_net)
    encoder = GaussianEncoder(encoder_net)

    # Build prior
    if args.prior_type == 'gaussian':
        prior = GaussianPrior(M)
    elif args.prior_type == 'mog':
        prior = MoGPrior(M, args.num_mixture_components)
    elif args.prior_type == 'vamp':
        # Use the same 'encoder' for the Vamp prior
        prior = VampPrior(encoder_net=encoder_net, K=10, input_shape=(1,28,28))
                          
    model = VAE(prior, decoder, encoder).to(device)

    # Run mode
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

    elif args.mode == 'test':
        model.load_state_dict(torch.load(args.model, map_location=args.device))
        test_elbo = evaluate_elbo(model, mnist_test_loader, device)
        # Summarize Data
        print(f"Test set ELBO: {test_elbo:.4f}")

        plot_latent_space(model, mnist_test_loader, device, args.latent_dim)
        plot_prior_samples(model, device, args.latent_dim, n_prior_samples=5000)
