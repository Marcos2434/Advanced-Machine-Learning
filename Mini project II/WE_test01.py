import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import grad

class Encoder(nn.Module):
    def __init__(self, input_dim=784, latent_dim=2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)  # Outputs latent representation
        )
    
    def forward(self, x):
        return self.fc(x)

# Compute the Jacobian of the encoder
def compute_jacobian(encoder, x):
    x = x.unsqueeze(0)  # Ensure batch dimension
    x.requires_grad_(True)
    z = encoder(x)
    jacobian = []
    for i in range(z.shape[1]):  # Iterate over latent dimensions
        grad_z = grad(z[0, i], x, retain_graph=True, create_graph=True)[0]
        jacobian.append(grad_z.squeeze().detach().numpy())
    return np.stack(jacobian, axis=0)

# Compute the latent space metric tensor G_z
def compute_metric_tensor(encoder, dataset):
    metric_tensors = []
    for x, _ in dataset:
        J_z = compute_jacobian(encoder, x)
        G_z = J_z.T @ J_z  # G_z = J_z^T J_z
        metric_tensors.append(G_z)
    return np.array(metric_tensors)

# Visualization function
def visualize_metric_tensor(metric_tensors):
    det_G = [np.linalg.det(G) for G in metric_tensors]
    plt.hist(det_G, bins=50, alpha=0.7, color='blue')
    plt.xlabel("Determinant of G_z")
    plt.ylabel("Frequency")
    plt.title("Distribution of Learned Latent Space Metric")
    plt.show()

# Example usage
torch.manual_seed(42)
encoder = Encoder()
dummy_data = [ (torch.randn(784), 0) for _ in range(1000) ]
metric_tensors = compute_metric_tensor(encoder, dummy_data)
visualize_metric_tensor(metric_tensors)
print('abc')
