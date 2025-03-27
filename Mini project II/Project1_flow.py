# Completed by:
# Tischler Ferreira, Martin Lautaro -- s240035
import torch
import torch.nn as nn
import torch.distributions as td
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm
import argparse
import numpy as np


class GaussianBase(nn.Module):
    def __init__(self, D):
        super(GaussianBase, self).__init__()
        self.D = D
        self.mean = nn.Parameter(torch.zeros(self.D), requires_grad=False)
        self.std = nn.Parameter(torch.ones(self.D), requires_grad=False)

    def forward(self):
        return td.Independent(td.Normal(loc=self.mean, scale=self.std), 1)


class MaskedCouplingLayer(nn.Module):
    def __init__(self, D, num_hidden, mask_type, layer_idx):
        super(MaskedCouplingLayer, self).__init__()

        self.D = D
        self.mask = self.init_mask(D, mask_type, layer_idx)

        self.scale_net = nn.Sequential(
            nn.Linear(D, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, D),
            nn.Tanh()  # For stability
        )

        self.translation_net = nn.Sequential(
            nn.Linear(D, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, D)
        )

    def init_mask(self, D, mask_type, layer_idx):
        if mask_type == 'random':
            mask = torch.randint(0, 2, (D,)).float()
        elif mask_type == 'chequerboard':
            side = int(np.sqrt(D))
            mask = torch.zeros((side, side))
            mask[::2, ::2] = 1
            mask[1::2, ::2] = 0
            mask[::2, 1::2] = 0
            mask[1::2, 1::2] = 1
            if layer_idx % 2 == 1:
                mask = 1 - mask
            mask = mask.flatten()
        else:
            raise ValueError(f"Unknown mask type: {mask_type}")
        return nn.Parameter(mask, requires_grad=False)

    def forward(self, z):
        masked_z = z * self.mask
        s = self.scale_net(masked_z)
        t = self.translation_net(masked_z)
        z_new = masked_z + (1 - self.mask) * (z * torch.exp(s) + t)
        log_det_J = ((1 - self.mask) * s).sum(-1)
        return z_new, log_det_J

    def inverse(self, x):
        masked_x = x * self.mask
        s = self.scale_net(masked_x)
        t = self.translation_net(masked_x)
        z = masked_x + (1 - self.mask) * ((x - t) * torch.exp(-s))
        log_det_J = -((1 - self.mask) * s).sum(-1)
        return z, log_det_J


class Flow(nn.Module):
    def __init__(self, base, transformations):
        super(Flow, self).__init__()
        self.base = base
        self.transformations = nn.ModuleList(transformations)

    def forward(self, z):
        sum_log_det_J = 0
        for T in self.transformations:
            z, log_det_J = T(z)
            sum_log_det_J += log_det_J
        return z, sum_log_det_J

    def inverse(self, x):
        sum_log_det_J = 0
        for T in reversed(self.transformations):
            x, log_det_J = T.inverse(x)
            sum_log_det_J += log_det_J
        return x, sum_log_det_J

    def log_prob(self, x):
        z, log_det_J = self.inverse(x)
        return self.base().log_prob(z) + log_det_J

    def sample(self, sample_shape=(1,)):
        z = self.base().sample(sample_shape)
        return self.forward(z)[0]

    def loss(self, x):
        return -self.log_prob(x).mean()


def train(model, optimizer, data_loader, epochs, device):
    model.train()
    progress_bar = tqdm(range(len(data_loader) * epochs), desc="Training")

    for epoch in range(epochs):
        for x, _ in data_loader:
            x = x.to(device)
            optimizer.zero_grad()
            loss = model.loss(x)
            loss.backward()
            optimizer.step()

            progress_bar.set_postfix(loss=f"{loss.item():.4f}", epoch=f"{epoch+1}/{epochs}")
            progress_bar.update()


def load_mnist(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x + torch.rand_like(x) / 255),
        transforms.Lambda(lambda x: x.flatten())
    ])
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data/', train=True, download=True, transform=transform),
        batch_size=batch_size, shuffle=True
    )
    return train_loader


def save_samples(model, device, save_path='project_flowbis_samples.png', n_samples=4):
    with torch.no_grad():
        samples = model.sample((n_samples,)).to(device)
        samples = samples.view(-1, 1, 28, 28)
        save_image(samples, save_path, nrow=8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'sample'])
    parser.add_argument('--model', default='flow_model.pt')
    parser.add_argument('--samples', default='samples.png')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num-transformations', type=int, default=8)
    parser.add_argument('--num-hidden', type=int, default=128)
    parser.add_argument('--mask-type', choices=['random', 'chequerboard'], default='chequerboard')

    args = parser.parse_args()

    device = torch.device(args.device)

    # Load dequantized & flattened MNIST
    train_loader = load_mnist(args.batch_size)

    # Define base distribution
    D = 28 * 28
    base = GaussianBase(D)

    # Build the flow
    transformations = [
        MaskedCouplingLayer(D, args.num_hidden, args.mask_type, layer_idx)
        for layer_idx in range(args.num_transformations)
    ]

    model = Flow(base, transformations).to(device)

    # Run Mode
    if args.mode == 'train':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        train(model, optimizer, train_loader, args.epochs, device)
        torch.save(model.state_dict(), args.model)

    elif args.mode == 'sample':
        model.load_state_dict(torch.load(args.model, map_location=device))
        save_samples(model, device, args.samples)


if __name__ == '__main__':
    main()
