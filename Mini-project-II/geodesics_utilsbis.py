#geodesics_utils.py
import torch
from torch.autograd.functional import jacobian
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import random
import os

def compute_decoder_jacobian(decoder, z):
    """
    Computes the Jacobian of the decoder mean at point z.
    
    Args:
        decoder: torch.nn.Module, maps z -> f(z) in R^D (e.g., GaussianDecoder)
        z: torch.Tensor of shape (M,) or (1, M), a single latent point
    
    Returns:
        J: Jacobian, torch.Tensor of shape (D, M), where D is the data dimension (e.g., 784 for MNIST)
    """
    z = z.detach().requires_grad_(True)  # Detach and enable gradient tracking
    
    def f_single_input(z_single):
        """
        Function to compute the decoder output for a single input z_single.
        """
        z_single = z_single.unsqueeze(0)  # Add batch dim: (1, M)
        decoded = decoder(z_single).mean  # Shape: (1, 1, 28, 28)
        return decoded.squeeze(0).reshape(-1)  # Shape: (D,), e.g., (784,)

    J = jacobian(f_single_input, z)  # Shape: (D, M) or (1, D, M) if z has batch dim
    if J.dim() == 3:
        J = J.squeeze(0)  # Remove batch dim if present
    return J  # Shape: (D, M)

def compute_pullback_metric(decoder, z):
    """
    Computes the pullback metric G_z = J_f(z)^T J_f(z).
    
    Args:
        decoder: Decoder network (e.g., GaussianDecoder)
        z: torch.Tensor of shape (M,) or (1, M), a single latent point
    
    Returns:
        G: torch.Tensor of shape (M, M), the pullback metric
    """
    J = compute_decoder_jacobian(decoder, z)  # Shape: (D, M)
    G = J.T @ J  # Shape: (M, M)
    return G

def compute_energy(decoder, curve_points):
    """
    Computes the discrete energy of a curve under the pullback metric for a single decoder.
    
    Args:
        decoder: Decoder network (e.g., GaussianDecoder), maps latent points to data space
        curve_points: torch.Tensor of shape (T, M), where T is the number of points along the curve,
                      and M is the latent dimension
    
    Returns:
        energy: torch.Tensor (scalar), the total energy of the curve
    """
    energy = torch.tensor(0.0, device=curve_points.device)
    dt = 1.0 / (len(curve_points) - 1)  # Segment length scaling factor
    
    for i in range(len(curve_points) - 1):
        z0 = curve_points[i]  # Shape: (M,)
        z1 = curve_points[i + 1]  # Shape: (M,)
        dz = z1 - z0  # Shape: (M,)
        
        # Compute pullback metric at the midpoint
        z_mid = (z0 + z1) / 2  # Shape: (M,)
        G = compute_pullback_metric(decoder, z_mid)  # Shape: (M, M)
        
        # Compute local energy: dz^T G dz, scaled by dt
        local_energy = dz.view(1, -1) @ G @ dz.view(-1, 1)  # Shape: (1, 1)
        local_energy = local_energy.squeeze() * dt  # Scalar, scaled by segment length
        energy = energy + local_energy
    
    return energy

def compute_length(decoder, curve):
    """
    Computes the approximate geodesic length of a curve via the pullback metric.
    
    Args:
        decoder: Decoder network (e.g., GaussianDecoder)
        curve: torch.Tensor of shape (T, M), where T is the number of points along the curve,
               and M is the latent dimension
    
    Returns:
        length: float, the total length of the curve
    """
    length = 0.0
    
    for i in range(len(curve) - 1):
        z1 = curve[i]  # Shape: (M,)
        z2 = curve[i + 1]  # Shape: (M,)
        delta = z2 - z1  # Shape: (M,)
        z_mid = (z1 + z2) / 2.0  # Shape: (M,)
        z_mid = z_mid.unsqueeze(0)  # Shape: (1, M)
        z_mid.requires_grad_(True)

        # Define a function to compute the decoder output as a flat vector
        def decoder_flat(z):
            return decoder(z).mean.view(-1)  # Shape: (D,), e.g., (784,)

        # Compute Jacobian: shape (D, M)
        J = torch.autograd.functional.jacobian(decoder_flat, z_mid).squeeze(1)  # Shape: (D, M)
        
        # Compute pullback metric G = J^T J: shape (M, M)
        G = J.transpose(0, 1) @ J  # Shape: (M, M)
        
        # Compute length increment: sqrt(delta^T G delta)
        length_increment = torch.sqrt(delta @ G @ delta).item()
        length += length_increment
    
    return length

# def plot_geodesic_latents(geodesic, ax=None, color='C0', show_endpoints=True, label_once=False):
#     """
#     Plot Geodesic in Latent Space
    
#     mainly a courtesy of chatgpt
#     """
#     if ax is None:
#         fig, ax = plt.subplots()

#     # geodesic is already a tensor of shape (num_points, M), no need to stack
#     points = geodesic.detach().cpu().numpy()  # Shape: (num_points, M)
#     z1, z2 = points[:, 0], points[:, 1]  # points of the geodesic in latent space

#     # Add labels only for first call if label_once=True
#     geodesic_label = 'Pullback geodesic' if not label_once else None
#     straight_label = 'Straight line' if not label_once else None

#     # Plot geodesic curve (solid)
#     ax.plot(z1, z2, '-', lw=2, color=color, label=geodesic_label)

#     # Plot straight line between endpoints (dashed)
#     ax.plot([z1[0], z1[-1]], [z2[0], z2[-1]], '--', color=color, alpha=0.5, label=straight_label)

#     # Optionally mark endpoints
#     if show_endpoints:
#         ax.plot(z1[0], z2[0], 'o', color=color)
#         ax.plot(z1[-1], z2[-1], 'x', color=color)

#     ax.set_xlabel('z1')
#     ax.set_ylabel('z2')
#     ax.set_title('Geodesic in Latent Space')
#     ax.axis('equal')
#     ax.grid(True)

def plot_geodesic_latents(geodesic, ax, color='blue', label_once=False, alpha=0.8):
    """
    Plot a geodesic curve in latent space with enhanced styling.
    """
    # Extract x and y coordinates
    x = geodesic[:, 0].numpy()
    y = geodesic[:, 1].numpy()
    
    # Plot the actual geodesic curve
    ax.plot(x, y, '-', color=color, linewidth=2.5, alpha=alpha, 
            label='Pullback geodesic' if not label_once else None)
    
    # Plot the straight line connecting start and end points
    ax.plot([x[0], x[-1]], [y[0], y[-1]], '--', color=color, linewidth=1.8, 
            alpha=0.5, label='Straight line' if not label_once else None)
    
    # Mark the start and end points
    ax.scatter([x[0]], [y[0]], color=color, s=50, marker='o', edgecolor='black', linewidth=0.8)
    ax.scatter([x[-1]], [y[-1]], color=color, s=50, marker='o', edgecolor='black', linewidth=0.8)

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

def plot_latent_std_background_across_decoders(decoders, grid_size=100, z1_range=(-6, 6), z2_range=(-6, 6), device='cpu', ax=None):
    """
    Plot background of std dev of decoder outputs across multiple decoders for the same latent points.
    
    Args:
        decoders: List of decoder networks (e.g., 10 decoders).
        grid_size: Size of the latent grid.
        z1_range, z2_range: Ranges for the latent dimensions.
        device: Device to perform computations on.
        ax: Matplotlib axis for plotting (optional).
    """
    if ax is None:
        fig, ax = plt.subplots()

    # Generate grid
    lin_z1 = torch.linspace(z1_range[0], z1_range[1], grid_size)
    lin_z2 = torch.linspace(z2_range[0], z2_range[1], grid_size)
    zz1, zz2 = torch.meshgrid(lin_z1, lin_z2, indexing='ij')
    grid = torch.stack([zz1.reshape(-1), zz2.reshape(-1)], dim=1).to(device)  # Shape: (N, 2), N = grid_size * grid_size

    num_decoders = len(decoders)
    N = grid_size * grid_size

    with torch.no_grad():
        # Decode grid with all decoders
        decoder_outputs = torch.stack([decoder(grid).mean for decoder in decoders], dim=1)  # Shape: (N, num_decoders, 1, 28, 28)
        decoder_outputs = decoder_outputs.squeeze(2)  # Shape: (N, num_decoders, 28, 28)

        # Compute standard deviation across decoders for each pixel, then mean over pixels
        std_across_decoders = decoder_outputs.std(dim=1)  # Shape: (N, 28, 28), std over num_decoders
        std_map = std_across_decoders.mean(dim=(1, 2)).cpu().numpy()  # Shape: (N,), mean std over pixels

    std_map = std_map.reshape(grid_size, grid_size)

    # Plot heatmap
    im = ax.imshow(
        std_map,
        origin='lower',
        extent=[z1_range[0], z1_range[1], z2_range[0], z2_range[1]],
        cmap='viridis',
        aspect='auto'
    )

    ax.set_xlim(z1_range)
    ax.set_ylim(z2_range)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Mean Std Dev Across Decoders')

    return ax

def compute_curve_energies(decoders, curve_points):
    """
    Compute energy of same curve under multiple decoders.
    
    Args:
        decoders: list of decoder networks (each from an ensemble)
        curve_points: list of torch.Tensors (latent points on curve)
    
    Returns:
        energies: list of scalar torch.Tensors (per-model energy)
        mean_energy: scalar (model-average energy)
        cov: scalar (coefficient of variation)
    """
    energies = []
    for decoder in decoders:
        energy = compute_energy(decoder, curve_points)
        energies.append(energy.item())  # detach scalar

    energies = torch.tensor(energies)
    mean_energy = torch.mean(energies)
    std_energy = torch.std(energies)
    cov = std_energy / mean_energy
    return energies, mean_energy, cov

def compute_length(decoder, curve):
    """
    Approximate geodesic length via pullback metric.
    Assumes `curve` is a (T, M) tensor of points in latent space.
    """
    length = 0.0
    for i in range(len(curve) - 1):
        z1 = curve[i]
        z2 = curve[i + 1]
        delta = z2 - z1
        z_mid = (z1 + z2) / 2.0
        z_mid = z_mid.unsqueeze(0)  # (1, M)
        z_mid.requires_grad_(True)

        # Make sure the decoder output is flat (D,)
        def decoder_flat(z):
            return decoder(z).mean.view(-1)  # (D,)

        # Compute Jacobian: shape (D, M)
        J = torch.autograd.functional.jacobian(decoder_flat, z_mid).squeeze(1)  # shape (D, M)
        #print(f"Step {i+1}/{len(curve)-1}")
        #print(f"  z_mid shape: {z_mid.shape}")
        #print(f"  J shape: {J.shape}")
        #print(f"  delta shape: {delta.shape}")
        # Compute pullback metric G = JᵀJ: shape (M, M)
        G = J.transpose(0, 1) @ J  # Use .transpose instead of .T to avoid warning
        #print(f"  G shape: {G.shape}")
        #print(f"  delta @ G @ delta: {(delta @ G @ delta).item():.4f}")
        # Compute length increment
        length += torch.sqrt(delta @ G @ delta).item()
    return length

def model_average_energy(c, decoders, num_samples=1):
    """
    Compute the model-average curve energy using Monte Carlo sampling, keeping gradients.
    
    Args:
        c: torch.Tensor of shape (N+1, M), the discretized curve in latent space.
        decoders: List of decoder models (ensemble members).
        num_samples: Number of Monte Carlo samples (decoder pairs) per segment.
    
    Returns:
        energy: torch.Tensor (scalar), the approximated model-average energy.
    """
    N = len(c) - 1  # Number of segments
    energy = torch.tensor(0.0, device=c.device)
    
    for i in range(N):
        segment_energy = torch.tensor(0.0, device=c.device)
        z_i = c[i].unsqueeze(0)      # Shape: (1, M)
        z_ip1 = c[i+1].unsqueeze(0)  # Shape: (1, M)
        
        # Monte Carlo sampling over decoder pairs
        for _ in range(num_samples):
            fl = random.choice(decoders)
            fk = random.choice(decoders)
            
            f1 = fl(z_i).mean.view(-1)    # Shape: (D,)
            f2 = fk(z_ip1).mean.view(-1)  # Shape: (D,)
            squared_diff = torch.norm(f1 - f2) ** 2
            segment_energy = segment_energy + squared_diff
        
        # Average over samples for this segment
        segment_energy = segment_energy / num_samples
        energy = energy + segment_energy
    
    return energy

def optimize_geodesic(z_start, z_end, num_points, num_iters, lr, energy_fn, convergence_threshold=1e-3, window_size=10, plot=False, check_convergence=False):
    device = z_start.device
    M = z_start.shape[0]
    
    t = torch.linspace(0, 1, num_points, device=device).unsqueeze(-1)
    z_start = z_start.unsqueeze(0)
    z_end = z_end.unsqueeze(0)
    curve = (1 - t) * z_start + t * z_end
    
    intermediate_points = curve[1:-1].clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([intermediate_points], lr=lr)
    
    energy_history = []
    curve_changes = []  # Track changes in the curve
    
    with tqdm(range(num_iters), desc="Optimizing geodesic") as pbar:
        for step in pbar:
            optimizer.zero_grad()
            curve = torch.cat([z_start, intermediate_points, z_end], dim=0)
            energy = energy_fn(curve)
            energy_history.append(energy.item())
            
            # Compute change in intermediate points
            if step > 0 and check_convergence:
                change = torch.norm(intermediate_points - prev_intermediate_points).item()
                curve_changes.append(change)
            prev_intermediate_points = intermediate_points.clone().detach()
            
            energy.backward()
            torch.nn.utils.clip_grad_norm_([intermediate_points], max_norm=1.0)
            optimizer.step()
            
            pbar.set_description(f"Optimizing geodesic, Energy: {energy.item():.4f}")
            
            if len(energy_history) >= window_size and check_convergence:
                recent_energies = energy_history[-window_size:]
                moving_avg = np.mean(recent_energies)
                if len(energy_history) >= 2 * window_size:
                    prev_energies = energy_history[-2*window_size:-window_size]
                    prev_moving_avg = np.mean(prev_energies)
                    if prev_moving_avg > 0:
                        relative_change = abs(moving_avg - prev_moving_avg) / prev_moving_avg
                        if relative_change < convergence_threshold:
                            print(f"Converged at step {step+1}/{num_iters}, "
                                  f"Relative change in moving average: {relative_change:.6f}")
                            break
    
    final_curve = torch.cat([z_start, intermediate_points.detach(), z_end], dim=0)
    final_energy = energy_fn(final_curve)
    print(f"Final energy after optimization: {final_energy.item():.4f}")
    
    # Plot energy history
    if plot:
        plt.figure(figsize=(8, 4))
        plt.plot(energy_history, label='Energy', alpha=0.5)
        if len(energy_history) >= window_size:
            moving_avg = np.convolve(energy_history, np.ones(window_size)/window_size, mode='valid')
            plt.plot(range(window_size-1, len(energy_history)), moving_avg, label='Moving Average', color='red')
        plt.xlabel('Iteration')
        plt.ylabel('Energy')
        plt.title('Energy During Geodesic Optimization')
        plt.grid(True)
        plt.legend()
        plt.savefig(f"energy_history_geodesic_{id(final_curve)}.png")
        plt.show()
    
        # Plot curve changes
        if curve_changes:
            plt.figure(figsize=(8, 4))
            plt.plot(curve_changes, label='Curve Change (L2 Norm)')
            plt.xlabel('Iteration')
            plt.ylabel('Change in Intermediate Points')
            plt.title('Curve Changes During Geodesic Optimization')
            plt.grid(True)
            plt.legend()
            plt.savefig(f"curve_changes_geodesic_{id(final_curve)}.png")
            plt.show()
    
    return final_curve

# Helper function to compute the metric determinant for a single decoder
def compute_metric_determinant(decoder, z_grid, device):
    """
    Compute the determinant of the pullback metric G at each point in z_grid.
    Decoder outputs an Independent Normal distribution.
    """
    z_grid = z_grid.to(device).requires_grad_(True)  # Enable gradients from the start
    decoder.eval()
    # Compute the distribution and extract the mean with gradients
    dist = decoder(z_grid)  # Independent Normal distribution
    outputs = dist.base_dist.loc  # Shape: (n_points, 1, 28, 28) for MNIST
    outputs_flat = outputs.view(outputs.size(0), -1)  # Flatten to (n_points, 784)

    # Compute the Jacobian in batches to reduce memory usage
    n_points, output_dim = outputs_flat.shape  # (n_points, 784)
    batch_size = 64  # Process 64 output dimensions at a time
    J = torch.zeros(n_points, output_dim, 2, device=device)  # Shape: (n_points, 784, 2)

    for start_idx in range(0, output_dim, batch_size):
        end_idx = min(start_idx + batch_size, output_dim)
        batch_size_current = end_idx - start_idx
        outputs_batch = outputs_flat[:, start_idx:end_idx]  # Shape: (n_points, batch_size_current)

        # Compute gradients for each output dimension in the batch
        grad_batch = torch.zeros(n_points, batch_size_current, 2, device=device)
        for i in range(batch_size_current):
            grad_outputs = torch.ones(n_points, device=device)  # Shape: (n_points,)
            grad = torch.autograd.grad(outputs_batch[:, i], z_grid, 
                                       grad_outputs=grad_outputs, 
                                       create_graph=False, 
                                       retain_graph=True)[0]  # Shape: (n_points, 2)
            grad_batch[:, i, :] = grad

        J[:, start_idx:end_idx, :] = grad_batch  # Shape: (n_points, batch_size_current, 2)
        # Clear intermediate tensors to free memory
        del grad_batch, outputs_batch
        torch.cuda.empty_cache()

    # Compute the pullback metric G = J^T J
    G = torch.bmm(J.transpose(1, 2), J)  # (n_points, 2, 784) @ (n_points, 784, 2) -> (n_points, 2, 2)
    det_G = torch.det(G).detach().cpu().numpy()  # Detach and convert to numpy
    z_grid.requires_grad_(False)  # Clean up
    del J, G
    torch.cuda.empty_cache()
    return det_G

# Helper function to compute standard deviation across ensemble decoders
def compute_ensemble_std(decoders, z_grid, device):
    z_grid = z_grid.to(device)
    outputs = []
    for decoder in decoders:
        decoder.eval()
        with torch.no_grad():  # No gradients needed for std computation
            dist = decoder(z_grid)  # Independent Normal distribution
            out = dist.base_dist.loc  # Use mean of the Normal distribution
            out_flat = out.view(out.size(0), -1)  # Flatten to (n_points, 784)
            outputs.append(out_flat.cpu().numpy())
    outputs = np.stack(outputs, axis=0)  # Shape: (num_decoders, n_points, 784)
    std = np.std(outputs, axis=0)  # Std across decoders: (n_points, 784)
    return std.mean(axis=1)  # Average std over pixels: (n_points,)

def plot_geodesics(latent_pairs, latent_pair_labels, geodesics, test_latents, test_labels, model, mode, num_decoders, device, experiment_folder, filename_suffix=""):
    """
    Plot geodesics in latent space with test points and latent pairs colored by class.
    
    Args:
        latent_pairs (list): List of tuples (z1, z2) representing the start and end points of geodesics.
        latent_pair_labels (list): List of tuples (label1, label2) with class labels for each latent pair.
        geodesics (list): List of geodesic paths, each a tensor of shape (n_steps, 2).
        test_latents (torch.Tensor): Latent projections of test points, shape (num_test_points, 2).
        test_labels (torch.Tensor): Class labels of test points, shape (num_test_points,).
        model: The model (single decoder or ensemble model) to compute the background.
        mode (str): Either "geodesics_single" or "geodesics_ensemble".
        num_decoders (int): Number of decoders (1 for single, >1 for ensemble).
        device (str): Device to perform computations on.
        experiment_folder (str): Folder to save the plot.
        filename_suffix (str): Suffix to append to the filename of the plot.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Compute dynamic limits based on the test_latents data
    z1_min, z1_max = test_latents[:, 0].min().item(), test_latents[:, 0].max().item()
    z2_min, z2_max = test_latents[:, 1].min().item(), test_latents[:, 1].max().item()

    # Add some padding for better visualization
    padding = 0.1  
    z1_range = (z1_min - padding * abs(z1_min), z1_max + padding * abs(z1_max))
    z2_range = (z2_min - padding * abs(z2_min), z2_max + padding * abs(z2_max))

    zlim = {'z1': z1_range, 'z2': z2_range}  # Updated limits
        
    # Compute the determinant of the metric G
    n_grid = 300
    z1_vals = np.linspace(zlim['z1'][0], zlim['z1'][1], n_grid)
    z2_vals = np.linspace(zlim['z2'][0], zlim['z2'][1], n_grid)
    Z1, Z2 = np.meshgrid(z1_vals, z2_vals)
    z_grid = torch.tensor(np.stack([Z1.ravel(), Z2.ravel()], axis=1), dtype=torch.float32)

    # Plot the background
    if mode == "geodesics_single":
        # Use plot_latent_std_background for single decoder
        # plot_latent_std_background(model.decoders[0], grid_size=100, z1_range=zlim['z1'], z2_range=zlim['z2'], device=device, ax=ax)
        det_G = compute_metric_determinant(model.decoders[0], z_grid, device)
        det_G = det_G.reshape(n_grid, n_grid)
        # Plot background as a heatmap (instead of contourf)
        im = ax.imshow(
            np.log1p(det_G),  # log1p for better scaling
            # det_G,
            origin='lower',
            extent=[zlim['z1'][0], zlim['z1'][1], zlim['z2'][0], zlim['z2'][1]],
            cmap='viridis',
            aspect='equal'
        )
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('log(1 + det(G))')
        # cbar.set_label('det(G)')
    elif mode == "geodesics_ensemble":
        # Compute standard deviation across decoders for ensemble
        ensemble_decoders_subset = model.decoders[:10]
        std_values = compute_ensemble_std(ensemble_decoders_subset, z_grid, device)
        std_values = std_values.reshape(n_grid, n_grid)
        im = ax.imshow(
            std_values,
            origin='lower',
            extent=[zlim['z1'][0], zlim['z1'][1], zlim['z2'][0], zlim['z2'][1]],
            cmap='viridis',
            aspect='equal'
        )
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Standard deviation of pixel values')

    # Scatter plot of all test points, colored by class
    colors = ['blue', 'orange', 'green']  # Colors for classes 0, 1, 2
    class_names = ['Class 0', 'Class 1', 'Class 2']
    for class_idx in range(3):  # Classes 0, 1, 2
        mask = test_labels == class_idx
        ax.scatter(
            test_latents[mask, 0], test_latents[mask, 1],
            s=10, alpha=0.3, color=colors[class_idx], label=class_names[class_idx], zorder=1
        )

    line_width = 2
    points_size = 50
    # Plot geodesics (solid black lines) and straight lines (dashed black lines)
    for i, geod in enumerate(geodesics):
        geod_cpu = geod.cpu()
        ax.plot(geod_cpu[:, 0], geod_cpu[:, 1], color='black', lw=line_width, zorder=3)  # Black solid lines for geodesics
        z1, z2 = latent_pairs[i]
        # ax.plot([z1[0].cpu(), z2[0].cpu()], [z1[1].cpu(), z2[1].cpu()], color='black', 
        #         lw=1.5, linestyle='--', alpha=0.7, zorder=3)  # Black dashed lines for straight paths

    # Plot latent pairs (start and end points) with colors based on their class
    for i, ((z1, z2), (label1, label2)) in enumerate(zip(latent_pairs, latent_pair_labels)):
        # Color z1 and z2 based on their class labels
        color1 = colors[label1.item()]
        color2 = colors[label2.item()]
        ax.scatter(z1[0].cpu(), z1[1].cpu(), s=points_size, color=color1, edgecolors='black', linewidth=line_width, zorder=4)  # Start point
        ax.scatter(z2[0].cpu(), z2[1].cpu(), s=points_size, color=color2, edgecolors='black', linewidth=line_width, zorder=4)  # End point


    # Customize plot
    ax.set_xlim(zlim['z1'])
    ax.set_ylim(zlim['z2'])
    ax.set_xlabel('z1', fontsize=12)
    ax.set_ylabel('z2', fontsize=12)
    ax.set_title(f"Geodesics in Latent Space with {num_decoders} Decoder{'s' if num_decoders > 1 else ''}", fontsize=14, pad=10)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(experiment_folder, f"geodesic_plot_{mode}_{filename_suffix}.png"), dpi=300)
    plt.show()
    print(f"Saved plot in {experiment_folder}/geodesic_plot_{mode}_{filename_suffix}.png")