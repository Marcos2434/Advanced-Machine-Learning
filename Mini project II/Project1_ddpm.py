# Completed by:
# Tischler Ferreira, Martin Lautaro -- s240035

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms, datasets
from tqdm import tqdm
import matplotlib.pyplot as plt
from unet import Unet  # Make sure unet.py is in the same folder

class DDPM(nn.Module):
    def __init__(self, network, beta_1=1e-4, beta_T=2e-2, T=1000):
        super(DDPM, self).__init__()
        self.network = network  # U-Net now
        self.T = T

        self.beta = nn.Parameter(torch.linspace(beta_1, beta_T, T), requires_grad=False)
        self.alpha = nn.Parameter(1 - self.beta, requires_grad=False)
        self.alpha_bar = nn.Parameter(torch.cumprod(self.alpha, dim=0), requires_grad=False)
        self.sqrt_alpha_bar = nn.Parameter(torch.sqrt(self.alpha_bar), requires_grad=False)
        self.sqrt_one_minus_alpha_bar = nn.Parameter(torch.sqrt(1 - self.alpha_bar), requires_grad=False)

    def sample_q(self, x0, t):
        noise = torch.randn_like(x0)
        mean = self.sqrt_alpha_bar[t][:, None] * x0
        std = self.sqrt_one_minus_alpha_bar[t][:, None]
        return mean + std * noise, noise

    def negative_elbo(self, x0):
        batch_size = x0.shape[0]
        t = torch.randint(0, self.T, (batch_size,), device=x0.device).long()

        xt, noise = self.sample_q(x0, t)
        t_norm = t.float().view(-1, 1) / self.T  # Normalize to [0, 1]

        predicted_noise = self.network(xt, t_norm)

        loss = F.mse_loss(predicted_noise, noise, reduction='none').sum(-1)
        return loss

    def sample(self, shape, device):
        x = torch.randn(shape, device=device)
        for t in reversed(range(self.T)):
            t_norm = torch.full((shape[0], 1), t / self.T, device=x.device)
            predicted_noise = self.network(x, t_norm)

            alpha_t = self.alpha[t]
            alpha_bar_t = self.alpha_bar[t]
            beta_t = self.beta[t]
            mean = (1 / torch.sqrt(alpha_t)) * (x - (beta_t / self.sqrt_one_minus_alpha_bar[t]) * predicted_noise)

            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)

            std = torch.sqrt(beta_t)
            x = mean + std * noise
        return x

    def loss(self, x):
        return self.negative_elbo(x).mean()

def load_mnist(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x + torch.rand_like(x) / 255),  # dequantize
        transforms.Lambda(lambda x: (x - 0.5) * 2),  # scale to [-1, 1]
        transforms.Lambda(lambda x: x.flatten())  # flatten to vector (28*28)
    ])
    train_data = datasets.MNIST('data/', train=True, download=True, transform=transform)
    return torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

def train(model, optimizer, data_loader, epochs, device):
    model.train()
    for epoch in range(epochs):
        for x, _ in tqdm(data_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            x = x.to(device)
            optimizer.zero_grad()
            loss = model.loss(x)
            loss.backward()
            optimizer.step()

def plot_samples(model, device, save_path="project_ddpm_mnist_unet_bis.png"):
    with torch.no_grad():
        samples = model.sample((4, 784), device=device).cpu()
        samples = (samples / 2 + 0.5).clamp(0, 1)  # Back to [0,1]
        samples = samples.view(-1, 1, 28, 28)

        torchvision.utils.save_image(samples, save_path, nrow=8)

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'sample'])
    parser.add_argument('--model', type=str, default='ddpm_mnist_unet.pt')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()

    device = torch.device(args.device)
    train_loader = load_mnist(args.batch_size)

    unet = Unet()
    model = DDPM(unet, T=1000).to(device)

    # Run Mode
    if args.mode == 'train':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        train(model, optimizer, train_loader, args.epochs, device)
        torch.save(model.state_dict(), args.model)

    elif args.mode == 'sample':
        model.load_state_dict(torch.load(args.model, map_location=device))
        plot_samples(model, device)

if __name__ == "__main__":
    main()
