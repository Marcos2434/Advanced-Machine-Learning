import torch
import torch.nn as nn
import torch.distributions as td
from tqdm import tqdm

class GaussianBase(nn.Module):
    def __init__(self, D):
        """
        Define a Gaussian base distribution with zero mean and unit variance.

                Parameters:
        M: [int] 
           Dimension of the base distribution.
        """
        super(GaussianBase, self).__init__()
        self.D = D
        self.mean = nn.Parameter(torch.zeros(self.D), requires_grad=False)
        self.std = nn.Parameter(torch.ones(self.D), requires_grad=False)

    def forward(self):
        """
        Return the base distribution.

        Returns:
        prior: [torch.distributions.Distribution]
        """
        return td.Independent(td.Normal(loc=self.mean, scale=self.std), 1)

class MaskedCouplingLayer(nn.Module):
    """
    An affine coupling layer for a normalizing flow.
    """

    def __init__(self, scale_net, translation_net, mask):
        """
        Define a coupling layer.

        Parameters:
        scale_net: [torch.nn.Module]
            The scaling network that takes as input a tensor of dimension `(batch_size, feature_dim)` and outputs a tensor of dimension `(batch_size, feature_dim)`.
        translation_net: [torch.nn.Module]
            The translation network that takes as input a tensor of dimension `(batch_size, feature_dim)` and outputs a tensor of dimension `(batch_size, feature_dim)`.
        mask: [torch.Tensor]
            A binary mask of dimension `(feature_dim,)` that determines which features (where the mask is zero) are transformed by the scaling and translation networks.
        """
        super(MaskedCouplingLayer, self).__init__()
        self.scale_net = scale_net
        self.translation_net = translation_net
        self.mask = nn.Parameter(mask, requires_grad=False)

    def forward(self, z):
        """
        Transform a batch of data through the coupling layer (from the base to data).

        Parameters:
        x: [torch.Tensor]
            The input to the transformation of dimension `(batch_size, feature_dim)`
        Returns:
        z: [torch.Tensor]
            The output of the transformation of dimension `(batch_size, feature_dim)`
        sum_log_det_J: [torch.Tensor]
            The sum of the log determinants of the Jacobian matrices of the forward transformations of dimension `(batch_size, feature_dim)`.
        """
        # Split input using the mask
        z_masked = z * self.mask  # Dimensions to keep unchanged
        
        # Pass unchanged dimensions through the networks
        s = self.scale_net(z_masked) * (1 - self.mask)  # Scale factor
        t = self.translation_net(z_masked) * (1 - self.mask)  # Translation factor
        
        # Apply the transformation: x = z_1 + (z_2 * exp(s) + t) where z_1 are masked dims
        # For masked dimensions (mask=1): x = z
        # For transformed dimensions (mask=0): x = z * exp(s) + t
        x = z_masked + (1 - self.mask) * (z * torch.exp(s) + t)
        
        # Log determinant of Jacobian: sum(s)
        # Only the transformed dimensions (mask=0) contribute to the Jacobian
        log_det_J = torch.sum(s, dim=1)
        
        return x, log_det_J
    
    def inverse(self, x):
        """
        Transform a batch of data through the coupling layer (from data to the base).

        Parameters:
        z: [torch.Tensor]
            The input to the inverse transformation of dimension `(batch_size, feature_dim)`
        Returns:
        x: [torch.Tensor]
            The output of the inverse transformation of dimension `(batch_size, feature_dim)`
        sum_log_det_J: [torch.Tensor]
            The sum of the log determinants of the Jacobian matrices of the inverse transformations.
        """
        # Split input using the mask
        x_masked = x * self.mask  # Dimensions that were kept unchanged
        
        # Pass unchanged dimensions through the networks
        s = self.scale_net(x_masked) * (1 - self.mask)  # Scale factor
        t = self.translation_net(x_masked) * (1 - self.mask)  # Translation factor
        
        # Apply the inverse transformation: z = (x - t) * exp(-s) for transformed dims
        # For masked dimensions (mask=1): z = x
        # For transformed dimensions (mask=0): z = (x - t) * exp(-s)
        z = x_masked + (1 - self.mask) * ((x - t) * torch.exp(-s))
        
        # Log determinant of Jacobian: -sum(s)
        # The negative sign comes from the inverse transformation
        log_det_J = -torch.sum(s, dim=1)
        
        return z, log_det_J


class Flow(nn.Module):
    def __init__(self, base, transformations):
        """
        Define a normalizing flow model.
        
        Parameters:
        base: [torch.distributions.Distribution]
            The base distribution.
        transformations: [list of torch.nn.Module]
            A list of transformations to apply to the base distribution.
        """
        super(Flow, self).__init__()
        self.base = base
        self.transformations = nn.ModuleList(transformations)

    def forward(self, z):
        """
        Transform a batch of data through the flow (from the base to data).
        
        Parameters:
        x: [torch.Tensor]
            The input to the transformation of dimension `(batch_size, feature_dim)`
        Returns:
        z: [torch.Tensor]
            The output of the transformation of dimension `(batch_size, feature_dim)`
        sum_log_det_J: [torch.Tensor]
            The sum of the log determinants of the Jacobian matrices of the forward transformations.            
        """
        sum_log_det_J = 0
        for T in self.transformations:
            x, log_det_J = T(z)
            sum_log_det_J += log_det_J
            z = x
        return x, sum_log_det_J
    
    def inverse(self, x):
        """
        Transform a batch of data through the flow (from data to the base).

        Parameters:
        x: [torch.Tensor]
            The input to the inverse transformation of dimension `(batch_size, feature_dim)`
        Returns:
        z: [torch.Tensor]
            The output of the inverse transformation of dimension `(batch_size, feature_dim)`
        sum_log_det_J: [torch.Tensor]
            The sum of the log determinants of the Jacobian matrices of the inverse transformations.
        """
        sum_log_det_J = 0
        for T in reversed(self.transformations):
            z, log_det_J = T.inverse(x)
            sum_log_det_J += log_det_J
            x = z
        return z, sum_log_det_J
    
    def log_prob(self, x):
        """
        Compute the log probability of a batch of data under the flow.

        Parameters:
        x: [torch.Tensor]
            The data of dimension `(batch_size, feature_dim)`
        Returns:
        log_prob: [torch.Tensor]
            The log probability of the data under the flow.
        """
        z, log_det_J = self.inverse(x)
        return self.base().log_prob(z) + log_det_J
    
    def sample(self, sample_shape=(1,)):
        """
        Sample from the flow.

        Parameters:
        n_samples: [int]
            Number of samples to generate.
        Returns:
        z: [torch.Tensor]
            The samples of dimension `(n_samples, feature_dim)`
        """
        z = self.base().sample(sample_shape)
        return self.forward(z)[0]
    
    def loss(self, x):
        """
        Compute the negative mean log likelihood for the given data bath.

        Parameters:
        x: [torch.Tensor] 
            A tensor of dimension `(batch_size, feature_dim)`
        Returns:
        loss: [torch.Tensor]
            The negative mean log likelihood for the given data batch.
        """
        return -torch.mean(self.log_prob(x))


def train(model, optimizer, data_loader, epochs, device):
    """
    Train a Flow model.

    Parameters:
    model: [Flow]
       The Flow model to train.
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
            if isinstance(x, list):
                x = x[0]
            x = x.to(device)
            optimizer.zero_grad()
            loss = model.loss(x)
            loss.backward()
            optimizer.step()

            # Update progress bar
            progress_bar.set_postfix(loss=f"⠀{loss.item():12.4f}", epoch=f"{epoch+1}/{epochs}")
            progress_bar.update()

# Masking strategy 1: Random initialized masking
def create_random_mask(D, seed=None):
    """
    Create a random binary mask with approximately half 0s and half 1s.
    
    Parameters:
    D: [int]
        Dimension of the mask.
    seed: [int]
        Random seed for reproducibility.
        
    Returns:
    mask: [torch.Tensor]
        A binary mask of dimension (D,)
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    # Create a random mask with approximately half 0s and half 1s
    mask = torch.randint(0, 2, (D,), dtype=torch.float)
    return mask

# Masking strategy 2: Checkerboard masking for 2D data
def create_checkerboard_mask(height, width):
    """
    Create a checkerboard mask for 2D data.
    
    Parameters:
    height: [int]
        Height of the 2D data.
    width: [int]
        Width of the 2D data.
        
    Returns:
    mask: [torch.Tensor]
        A binary checkerboard mask of dimension (height*width,)
    """
    # Create a grid of indices
    x, y = torch.meshgrid(torch.arange(height), torch.arange(width))
    
    # Create a checkerboard pattern: 1 where (x+y) is even, 0 where (x+y) is odd
    checkerboard = ((x + y) % 2 == 0).float()
    
    # Flatten to 1D
    mask = checkerboard.reshape(-1)
    
    return mask

def build_flow_with_random_masks(D, num_transformations, num_hidden, device='cpu'):
    """
    Build a flow model with random masks for each transformation.
    
    Parameters:
    D: [int]
        Dimension of the data.
    num_transformations: [int]
        Number of transformations to use.
    num_hidden: [int]
        Number of hidden units in the neural networks.
    device: [str]
        Device to use for training.
        
    Returns:
    model: [Flow]
        The flow model.
    """
    # Define prior distribution
    base = GaussianBase(D)
    
    # Define transformations with random masks
    transformations = []
    for i in range(num_transformations):
        # Create a random mask - using different seeds for each layer to ensure they are different
        mask = create_random_mask(D, seed=i)

        # Define neural networks for scaling and translation
        # scale_net = nn.Sequential(nn.Linear(D, num_hidden), nn.ReLU(), nn.Linear(num_hidden, D), nn.Tanh())
        scale_net = nn.Sequential(nn.Linear(D, num_hidden), nn.ReLU(), nn.Linear(num_hidden, num_hidden), nn.ReLU(), nn.Linear(num_hidden, D), nn.Tanh())
        translation_net = nn.Sequential(nn.Linear(D, num_hidden), nn.ReLU(), nn.Linear(num_hidden, D))
 
        
        # Add the coupling layer
        transformations.append(MaskedCouplingLayer(scale_net, translation_net, mask))
    
    # Define flow model
    model = Flow(base, transformations).to(device)
    
    return model

def build_flow_with_checkerboard_masks(D, height, width, num_transformations, num_hidden, device='cpu'):
    """
    Build a flow model with checkerboard masks, alternating the mask for each transformation.
    
    Parameters:
    D: [int]
        Dimension of the data.
    height: [int]
        Height of the 2D data.
    width: [int]
        Width of the 2D data.
    num_transformations: [int]
        Number of transformations to use.
    num_hidden: [int]
        Number of hidden units in the neural networks.
    device: [str]
        Device to use for training.
        
    Returns:
    model: [Flow]
        The flow model.
    """
    # Define prior distribution
    base = GaussianBase(D)
    
    # Create initial checkerboard mask
    mask = create_checkerboard_mask(height, width)
    
    # Define transformations with alternating checkerboard masks
    transformations = []
    for i in range(num_transformations):
        # Define neural networks for scaling and translation
        scale_net = nn.Sequential(nn.Linear(D, num_hidden), nn.ReLU(), nn.Linear(num_hidden, D), nn.Tanh())
        # scale_net = nn.Sequential(nn.Linear(D, num_hidden), nn.ReLU(), nn.Linear(num_hidden, num_hidden), nn.ReLU(), nn.Linear(num_hidden, D), nn.Tanh())
        translation_net = nn.Sequential(nn.Linear(D, num_hidden), nn.ReLU(), nn.Linear(num_hidden, D))
 
        
        # Add the coupling layer
        transformations.append(MaskedCouplingLayer(scale_net, translation_net, mask))
        
        # Invert the mask for the next layer (1-mask flips 0s to 1s and 1s to 0s)
        mask = 1 - mask
    
    # Define flow model
    model = Flow(base, transformations).to(device)
    
    return model


if __name__ == "__main__":
    import torch.utils.data
    from torchvision import datasets, transforms
    from torchvision.utils import save_image
    import ToyData
    import argparse
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, default='train', choices=['train', 'sample'], help='what to do when running the script (default: %(default)s)')
    parser.add_argument('--data', type=str, default='tg', choices=['tg', 'cb', 'mnist'], help='toy dataset to use {tg: two Gaussians, cb: chequerboard} (default: %(default)s)')
    parser.add_argument('--model', type=str, default='model.pt', help='file to save model to or load model from (default: %(default)s)')
    parser.add_argument('--samples', type=str, default='samples.png', help='file to save samples in (default: %(default)s)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], help='torch device (default: %(default)s)')
    parser.add_argument('--batch-size', type=int, default=10000, metavar='N', help='batch size for training (default: %(default)s)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N', help='number of epochs to train (default: %(default)s)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='V', help='learning rate for training (default: %(default)s)')
    parser.add_argument('--mask-type', type=str, default='random', choices=['random', 'checkerboard'], 
                        help='masking strategy to use (default: %(default)s)')

    args = parser.parse_args()
    print('# Options')
    for key, value in sorted(vars(args).items()):
        print(key, '=', value)

    # Generate the data
    if args.data == 'mnist':
        transform = transforms.Compose([transforms.ToTensor(),
                                      transforms.Lambda(lambda x: x + torch.rand(x.shape)/255),
                                    #   transforms.Normalize((0.5,), (0.5,)),  # Normalize to [-1, 1]
                                      transforms.Lambda(lambda x: x.flatten())])
        train_data = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
        test_data = datasets.MNIST(root="./data", train=False, transform=transform, download=True)
        train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False)

        D = next(iter(train_loader))[0].shape[1]
        height, width = 28, 28  # MNIST is 28x28
    
    else:
        n_data = 10000000
        toy = {'tg': ToyData.TwoGaussians, 'cb': ToyData.Chequerboard}[args.data]()
        train_loader = torch.utils.data.DataLoader(toy().sample((n_data,)), batch_size=args.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(toy().sample((n_data,)), batch_size=args.batch_size, shuffle=True)

        D = next(iter(train_loader)).shape[1]
        height, width = 1, D  # For 2D toy data, set arbitrary height and width
    
    # Parameters for the networks
    num_transformations = 10
    num_hidden = 1024
    
    # Build the flow model based on the chosen masking strategy
    if args.mask_type == 'random':
        model = build_flow_with_random_masks(D, num_transformations, num_hidden, args.device)
    else:  # checkerboard
        model = build_flow_with_checkerboard_masks(D, height, width, num_transformations, num_hidden, args.device)

    # Choose mode to run
    if args.mode == 'train':
        # Define optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        # Train model
        train(model, optimizer, train_loader, args.epochs, args.device)

        # Save model
        torch.save(model.state_dict(), args.model)

    elif args.mode == 'sample':
        import matplotlib.pyplot as plt
        import numpy as np

        model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))
        
        if args.data == 'mnist':
            # Generate samples
            num_samples = 100

            model.eval()
            with torch.no_grad():
                samples = (model.sample((num_samples,))).cpu() 

            # Transform the samples back to the original space
            # samples = samples * 0.5 + 0.5
            samples = samples.reshape(-1, 1, 28, 28)

            fig = plt.figure(figsize=(8, 8))
            columns = 10
            rows = 10
            for i in range(1, columns * rows + 1):
                img = samples[i - 1].cpu().detach().numpy().transpose(1, 2, 0).squeeze()
                fig.add_subplot(rows, columns, i)
                plt.axis('off')
                plt.imshow(img, cmap='gray')
            plt.savefig(args.samples)
            plt.close()
        else:
            # Generate samples
            model.eval()
            with torch.no_grad():
                samples = (model.sample((10000,))).cpu() 

            # Plot the density of the toy data and the model samples
            coordinates = [[[x,y] for x in np.linspace(*toy.xlim, 1000)] for y in np.linspace(*toy.ylim, 1000)]
            prob = torch.exp(toy().log_prob(torch.tensor(coordinates)))

            fig, ax = plt.subplots(1, 1, figsize=(7, 5))
            im = ax.imshow(prob, extent=[toy.xlim[0], toy.xlim[1], toy.ylim[0], toy.ylim[1]], origin='lower', cmap='YlOrRd')
            ax.scatter(samples[:, 0], samples[:, 1], s=1, c='black', alpha=0.5)
            ax.set_xlim(toy.xlim)
            ax.set_ylim(toy.ylim)
            ax.set_aspect('equal')
            fig.colorbar(im)
            plt.savefig(args.samples)
            plt.close()