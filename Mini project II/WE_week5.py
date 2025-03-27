import numpy as np
from scipy.integrate import quad
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as dist

# --- Exercise 5.5: Compute curve length numerically ---
def curve(t):
    """Defines the curve c(t) = (2t + 1, -t^2)"""
    return np.array([2 * t + 1, -t**2])

def speed_function(t):
    """Computes the speed |dc/dt| at t."""
    dc_dt = np.array([2, -2 * t])
    return np.linalg.norm(dc_dt)

def curve_length_numeric():
    """Computes curve length by integrating the speed function."""
    length, _ = quad(speed_function, 0, 1)
    return length

print("Numerical curve length:", curve_length_numeric())

# --- Exercise 5.6: Bernoulli VAE Training ---
class BernoulliVAE(nn.Module):
    def __init__(self, latent_dim=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim * 2)  # Mean and log variance
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x_flat = x.view(x.size(0), -1)
        q_params = self.encoder(x_flat)
        mu, logvar = torch.chunk(q_params, 2, dim=1)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

# Training loop (simplified for demonstration)
def train_vae(vae, dataloader, epochs=5, lr=1e-3):
    optimizer = optim.Adam(vae.parameters(), lr=lr)
    loss_fn = nn.BCELoss(reduction='sum')
    for epoch in range(epochs):
        for x, _ in dataloader:
            x = x.view(x.size(0), -1)
            x_recon, mu, logvar = vae(x)
            recon_loss = loss_fn(x_recon, x)
            kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + kl_div
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    print("Training complete.")
