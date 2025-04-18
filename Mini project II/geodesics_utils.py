import torch
from torch.autograd.functional import jacobian
import matplotlib.pyplot as plt
from tqdm import tqdm

def compute_decoder_jacobian(decoder, z):
    """
    Computes the Jacobian of the decoder mean at point z.
    Args:
        decoder: torch.nn.Module, maps z -> f(z) in R^D
        z: torch.Tensor of shape (M,) or (1, M)
    Returns:
        J: jacobian, torch.Tensor of shape (D, M)
    """
    
    # create a new tensor with the same data as z but detached from 
    # the computation graph such that any operations performed on it won't be tracked 
    z = z.detach().requires_grad_(True) 
    
    def f_single_input(z_single):
        """
        Function to compute the decoder output for a single input z_single
        """
        z_single = z_single.unsqueeze(0)  # Add batch dim: (*1*, M) so that it can be input to the decoder
        decoded = decoder(z_single).mean  # extract the .mean from the distribution output. shape: (1, 1, 28, 28)
        return decoded.squeeze(0).reshape(-1)  # shape: (784,)

    J = jacobian(f_single_input, z) # we take the jacobian of the decoder: f_single_input at input z: J(z)
    if J.dim() == 3: J = J.squeeze(0) # If J has an extra batch dimension, remove it.
    return J  # shape: (D, M)


def compute_pullback_metric(decoder, z):
    """
    Computes Pullback Metric G_z = J_f(z)^T J_f(z)
    Args:
        decoder: decoder network
        z: torch.Tensor of shape (M,) or (1, M)
    Returns:
        G: torch.Tensor of shape (M, M)
    """
    # from assignment: the pullback metric should be associated 
    # with the mean of the Gaussian decoder
    J = compute_decoder_jacobian(decoder, z)  # (D, M)
    G = J.T @ J  # (M, M)
    return G



from tqdm import tqdm

def compute_energy(decoder, curve_points):
    """
    Computes discrete energy of a curve under pullback metric.
    Args:
        decoder: decoder network
        curve_points: list of tensors, each shape (M,)
    Returns:
        scalar torch.Tensor
    """
    energy = 0.0
    for i in range(len(curve_points) - 1):
        z0 = curve_points[i]
        z1 = curve_points[i + 1]
        dz = z1 - z0
        G = compute_pullback_metric(decoder, (z0 + z1) / 2)
        local_energy = dz.view(1, -1) @ G @ dz.view(-1, 1) 
        local_energy = local_energy.squeeze()
        energy += local_energy
    return energy


def optimize_geodesic(decoder, z_start, z_end, num_points=10, num_iters=500, lr=1e-2):
    """
    Optimize Geodesic via Energy Minimization
    Compute geodesic from z_start to z_end using energy minimization.
    Args:
        decoder: decoder network
        z_start, z_end: torch.Tensor of shape (M,)
        num_points: number of total points in curve
        num_iters: gradient descent steps
        lr: learning rate
    Returns:
        List of torch.Tensor geodesic points
    """
    M = z_start.shape[-1]
    
    # Initialize intermediate points
    with torch.no_grad(): 
        # Initial guess: straight line from z_start to z_end
        inits = [z_start + (i / (num_points - 1)) * (z_end - z_start) for i in range(num_points)]
    intermediates = [torch.nn.Parameter(p.clone()) for p in inits[1:-1]] # exclude endpoints
    
    optimizer = torch.optim.Adam(intermediates, lr=lr) # optimizer for the intermediate points

    # Find geodesic with minimum energy
    for _ in tqdm(range(num_iters), desc="Optimizing geodesic"):
        optimizer.zero_grad()
        curve = [z_start] + intermediates + [z_end] # make the curve by concatenating all points
        energy = compute_energy(decoder, curve) # energy is differentiable w.r.t. intermediate points
        
        # compute the gradient of the energy
        energy.backward() 
        
        # take the step in the negative gradient direction
        optimizer.step()

    with torch.no_grad():
        curve = [z_start] + [p.clone() for p in intermediates] + [z_end]
        final_energy = compute_energy(decoder, curve)
        print(f"Final energy after optimization: {final_energy.item():.6f}")
    
    return curve


def plot_geodesic_latents(geodesic, ax=None, color='C0', show_endpoints=True, label_once=False):
    """
    Plot Geodesic in Latent Space
    
    mainly a courtesy of chatgpt
    """
    if ax is None:
        fig, ax = plt.subplots()

    points = torch.stack(geodesic).detach().cpu().numpy()
    z1, z2 = points[:, 0], points[:, 1] # points of the geodesic in latent space

    # Add labels only for first call if label_once=True
    geodesic_label = 'Pullback geodesic' if not label_once else None
    straight_label = 'Straight line' if not label_once else None

    # Plot geodesic curve (solid)
    ax.plot(z1, z2, '-', lw=2, color=color, label='Pullback geodesic')

    # Plot straight line between endpoints (dashed)
    ax.plot([z1[0], z1[-1]], [z2[0], z2[-1]], '--', color=color, alpha=0.5, label='Straight line')

    # Optionally mark endpoints
    if show_endpoints:
        ax.plot(z1[0], z2[0], 'o', color=color)
        ax.plot(z1[-1], z2[-1], 'x', color=color)

    ax.set_xlabel('z1')
    ax.set_ylabel('z2')
    ax.set_title('Geodesic in Latent Space')
    ax.axis('equal')
    ax.grid(True)
    #plt.show()
    #if ax is None:
    #    plt.legend()
    #    plt.show()
    ### This one works
    """
    Plot geodesic path in 2D latent space.
    points = torch.stack(geodesic).detach().cpu().numpy()
    plt.plot(points[:, 0], points[:, 1], marker='o')
    plt.title("Latent Space Geodesic")
    plt.axis('equal')
    plt.grid(True)
    plt.show()"""


def plot_decoded_images(decoder, geodesic):
    """
    Decodes and plots images along geodesic.
    
    mainly a courtesy of chatgpt
    """
    with torch.no_grad():
        imgs = [decoder(z.unsqueeze(0)).mean.view(28, 28).cpu() for z in geodesic]

    fig, axes = plt.subplots(1, len(imgs), figsize=(15, 2))
    for ax, img in zip(axes, imgs):
        ax.imshow(img, cmap='gray')
        ax.axis('off')
    plt.show()

def plot_latent_std_background(decoder, grid_size=100, zlim=6, device='cpu', ax=None):
    """
    Plot background of std dev of decoder output over a latent grid.
    
    mainly a courtesy of chatgpt
    """
    if ax is None:
        fig, ax = plt.subplots()

    # Generate grid
    lin = torch.linspace(-zlim, zlim, grid_size)
    zz1, zz2 = torch.meshgrid(lin, lin, indexing='ij')
    grid = torch.stack([zz1.reshape(-1), zz2.reshape(-1)], dim=1).to(device)  # shape: (grid_size**2, 2)

    with torch.no_grad():
        out = decoder(grid).mean  # (N, 1, 28, 28)
        std_map = out.std(dim=(1, 2, 3)).cpu().numpy()  # compute std over pixels

    std_map = std_map.reshape(grid_size, grid_size)

    # Plot heatmap
    im = ax.imshow(
        std_map.T,
        origin='lower',
        extent=[-zlim, zlim, -zlim, zlim],
        cmap='viridis',
        aspect='auto'
    )

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Standard deviation of pixel values')

    return ax

########################################
## New implementation of the plotting ##
########################################
def plot_latent_std_background(decoder, grid_size=100, z1_range=(-6, 6), z2_range=(-6, 6), device='cpu', ax=None):
    """
    Plot background of std dev of decoder output over a latent grid.
    """
    if ax is None:
        fig, ax = plt.subplots()

    # Generate grid
    lin_z1 = torch.linspace(z1_range[0], z1_range[1], grid_size)
    lin_z2 = torch.linspace(z2_range[0], z2_range[1], grid_size)
    zz1, zz2 = torch.meshgrid(lin_z1, lin_z2, indexing='ij')
    grid = torch.stack([zz1.reshape(-1), zz2.reshape(-1)], dim=1).to(device)


    with torch.no_grad():
        out = decoder(grid).mean  # (N, 1, 28, 28)
        std_map = out.std(dim=(1, 2, 3)).cpu().numpy()  # compute std over pixels

    std_map = std_map.reshape(grid_size, grid_size)

    # Plot heatmap
    im = ax.imshow(
        std_map,
        origin='lower',
        extent=[z1_range[0], z1_range[1], z2_range[0], z2_range[1]],
        cmap='viridis',
        aspect='auto'  # allow full fill of plot area
    )

    ax.set_xlim(z1_range)
    ax.set_ylim(z2_range)
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Standard deviation of pixel values')

    return ax

def plot_geodesic_latents(geodesic, ax=None, color='C0', show_endpoints=True, label_once=False):
    if ax is None:
        fig, ax = plt.subplots()

    points = torch.stack(geodesic).detach().cpu().numpy()
    z1, z2 = points[:, 0], points[:, 1]

    # Add labels only for first call if label_once=True
    geodesic_label = 'Pullback geodesic' if not label_once else None
    straight_label = 'Straight line' if not label_once else None

    # Plot geodesic curve (solid)
    ax.plot(z1, z2, '-', lw=2, color=color, label='Pullback geodesic')

    # Plot straight line between endpoints (dashed)
    ax.plot([z1[0], z1[-1]], [z2[0], z2[-1]], '--', color=color, alpha=0.5, label='Straight line')

    # Optionally mark endpoints
    if show_endpoints:
        ax.plot(z1[0], z2[0], 'o', color=color)
        ax.plot(z1[-1], z2[-1], 'x', color=color)

    ax.set_xlabel('z1')
    ax.set_ylabel('z2')
    ax.set_title('Geodesic in Latent Space')
    ax.axis('equal')
    ax.grid(False)