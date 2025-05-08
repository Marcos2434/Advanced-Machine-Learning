#!/usr/bin/env python3
# Mini Project 3 - Graph Generation Models
# Modified May 2025

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_networkx, to_dense_adj
from torch_geometric.nn import GCNConv, global_mean_pool
from networkx.algorithms.graph_hashing import weisfeiler_lehman_graph_hash as wl
import itertools


# Convert PyG datasets to networkx graphs
def graphs_from_dataset(ds):
    out = []
    
    for d in ds:
        g = to_networkx(d, to_undirected=True)
        g.remove_edges_from(nx.selfloop_edges(g))
        out.append(g)
        
    return out


# ===== SETUP =====
# Use CUDA if available, otherwise CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
rng = torch.Generator().manual_seed(0)

# Load dataset
dataset = TUDataset(root='./data/', name='MUTAG')
train_ds, val_ds, test_ds = torch.utils.data.random_split(
    dataset, (100, 44, 44), generator=rng
)
train_ld = DataLoader(train_ds, batch_size=32, shuffle=True)

# Convert to networkx format for analysis
train_nx = graphs_from_dataset(train_ds)
val_nx = graphs_from_dataset(val_ds)

# Calculate dataset statistics
all_degs = [d for G in train_nx for _, d in G.degree()]
MAX_DEG = float(np.percentile(all_degs, 95))
N_MAX = max(d.num_nodes for d in dataset)       # 28 for MUTAG
LATENT = 128  # latent dimension
HIDDEN = 256  # hidden layer width
LR = 1e-4     # learning rate
GAN_EPOCHS = 10000  # GAN training steps


# ===== BASELINE MODEL =====
def sample_er_connected(n_graph, ref_ds):
    """Generate connected Erdős–Rényi graphs with similar properties to reference dataset"""
    # Get node distribution from data
    num_nodes = np.array([d.num_nodes for d in ref_ds])
    vals, cnt = np.unique(num_nodes, return_counts=True)
    pN = cnt/cnt.sum()

    outs = []
    for _ in range(n_graph):
        # Sample node count according to empirical distribution
        N = int(np.random.choice(vals, p=pN))
        
        # Calculate typical edge density for this node count
        edges = [d.num_edges for d in ref_ds if d.num_nodes == N]
        dens = np.mean([e/(N*(N-1)/2) for e in edges])
        
        # Generate connected graph
        while True:
            g = nx.erdos_renyi_graph(N, dens)
            if nx.is_connected(g): 
                break
                
        outs.append(g)
    return outs


# ===== GRAPH VAE =====
class GraphVAE(nn.Module):
    """
    Graph Variational Autoencoder
    - GCN-based encoder maps graphs to latent distribution
    - MLP decoder reconstructs adjacency matrix
    """
    def __init__(self, in_dim, hid_dim, lat_dim, N_max):
        super().__init__()
        self.N_max = N_max
        self.lat_dim = lat_dim

        # Encoder layers
        self.enc1 = GCNConv(in_dim, hid_dim)
        self.enc2 = GCNConv(hid_dim, hid_dim)
        self.mu = nn.Linear(hid_dim, lat_dim)
        self.logv = nn.Linear(hid_dim, lat_dim)

        # Decoder - MLP that outputs edge probabilities
        self.dec = nn.Sequential(
            nn.Linear(lat_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, N_max * N_max)   # Flattened adjacency matrix
        )

    def encode(self, x, edge_index, batch):
        # GCN layers
        h = F.relu(self.enc1(x, edge_index))
        h = F.relu(self.enc2(h, edge_index))
        
        # Pool node features to graph representation
        g = global_mean_pool(h, batch)
        
        # Get distribution parameters
        mu, logv = self.mu(g), self.logv(g)
        
        # Sample via reparameterization trick
        std = torch.exp(0.5 * logv)
        z = mu + std * torch.randn_like(std)
        
        return z, mu, logv

    def decode(self, z, N):
        """Convert latent vector to symmetric adjacency matrix"""
        B = z.size(0)
        # Reshape to square matrix
        logits = self.dec(z).view(B, self.N_max, self.N_max)
        
        # Make symmetric by averaging with transpose
        logits = (logits + logits.transpose(1, 2)) / 2
        
        # Remove self-loops by masking diagonal
        mask = torch.eye(self.N_max, device=z.device).bool()
        logits = logits.masked_fill(mask, -9e15)
        
        # Convert to probabilities
        probs = torch.sigmoid(logits)
        
        # Return only relevant submatrix
        return probs[:, :N, :N]

    def forward(self, data):
        # Encode graph to latent space
        z, mu, logv = self.encode(data.x, data.edge_index, data.batch)
        
        # Decode to adjacency matrix
        N = data.num_nodes                      
        adj_hat = self.decode(z, N)
        
        return adj_hat, mu, logv


def train_vae(loader, in_dim=7, hid=128, lat=64, N_max=N_MAX, epochs=5000):
    """Train the VAE model"""
    vae = GraphVAE(in_dim, hid, lat, N_max).to(device)
    opt = torch.optim.Adam(vae.parameters(), 1e-3)
    
    for ep in range(epochs):
        for batch in loader:
            batch = batch.to(device)
            
            # Get true adjacency matrix
            adj_true = to_dense_adj(
                batch.edge_index,
                batch=batch.batch,
                max_num_nodes=N_max
            )
            
            # Get VAE reconstruction and latent parameters
            adj_hat, mu, logv = vae(batch)
            
            # Reconstruction loss
            loss_recon = F.binary_cross_entropy(adj_hat, adj_true)
            
            # KL divergence
            loss_kl = 0.5 * torch.mean(mu.pow(2) + logv.exp() - 1 - logv)
            
            # Combined loss with annealed KL term
            loss = loss_recon + 0.6 * loss_kl
            
            # Update
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        # Occasional progress report
        if ep == 1 or ep % 200 == 0:
            print(f"VAE epoch {ep:>4}/{epochs}   loss {loss.item():.3f} ")
            
    return vae


@torch.no_grad()
def sample_vae(vae, num_graphs, ref_ds):
    """Sample graphs from VAE latent space"""
    # Get node count distribution
    num_nodes = np.array([d.num_nodes for d in ref_ds])
    vals, cnt = np.unique(num_nodes, return_counts=True)
    pN = cnt / cnt.sum()

    out = []
    while len(out) < num_graphs:
        # Sample node count
        N = int(np.random.choice(vals, p=pN))
        
        # Sample from latent space
        z = torch.randn(1, vae.lat_dim, device=device)
        
        # Decode to edge probabilities
        probs = vae.decode(z, N)[0]
        
        # Sample binary adjacency by thresholding
        tri = (torch.rand_like(probs).triu(1) < probs.triu(1))
        A = (tri | tri.T).int().cpu().numpy()
        
        # Create networkx graph
        G = nx.from_numpy_array(A)
        
        # Keep only if connected
        if nx.is_connected(G):
            out.append(G)
            
    return out


# ===== GRAPH GAN =====

# Helpers for adjacency matrices
def pyg_graph_to_dense(d, n_max=N_MAX):
    """Convert PyG graph to dense adjacency with padding"""
    A = np.zeros((n_max, n_max), dtype=np.float32)
    A[d.edge_index[0], d.edge_index[1]] = 1.
    A[np.diag_indices_from(A)] = 0.  # No self-loops
    return A


class Generator(nn.Module):
    """Generator network that creates graph adjacency matrices"""
    def __init__(self, z_dim=LATENT, hidden=HIDDEN, n_max=N_MAX):
        super().__init__()
        out_len = n_max * n_max
        
        # MLP to transform noise to adjacency logits
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden, out_len),
        )
        self.n_max = n_max

    def forward(self, z):
        # Generate flattened adjacency
        logits = self.net(z)
        
        # Reshape to matrix
        logits = logits.view(-1, self.n_max, self.n_max)
        
        # Ensure symmetry
        logits = (logits + logits.transpose(1, 2)) / 2
        
        # Prevent self-loops
        mask = torch.eye(self.n_max, device=logits.device).bool()
        logits = logits.masked_fill(mask, -9e15)
        
        return logits


class Discriminator(nn.Module):
    """Discriminator network to classify real/fake graphs"""
    def __init__(self, hidden=HIDDEN, n_max=N_MAX):
        super().__init__()
        inp_len = n_max * n_max
        
        # MLP to classify adjacency matrices
        self.net = nn.Sequential(
            nn.Linear(inp_len, hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden, 1)
        )

    def forward(self, adj):
        # Flatten adjacency matrix
        flat = adj.view(adj.size(0), -1)
        
        # Classify as real/fake
        return self.net(flat)


def densify_batch(batch, N_max=N_MAX):
    """Convert PyG batch to dense adjacency matrices"""
    return to_dense_adj(
        batch.edge_index, 
        batch=batch.batch,
        max_num_nodes=N_max
    ).float()


def train_gan(loader, epochs=GAN_EPOCHS, latent_dim=LATENT, 
              hidden_dim=HIDDEN, n_max=N_MAX, lr=LR):
    """Train GAN with Wasserstein loss and gradient penalty"""
    # Initialize models
    G = Generator(z_dim=latent_dim, hidden=hidden_dim, n_max=n_max).to(device)
    D = Discriminator(hidden=hidden_dim, n_max=n_max).to(device)

    # Optimizers with momentum
    opt_g = torch.optim.Adam(G.parameters(), lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr, betas=(0.5, 0.999))

    # Helper functions
    one = lambda B: torch.ones(B, 1, device=device)
    zero = lambda B: torch.zeros(B, 1, device=device)

    # Setup training loop
    data_iter = itertools.cycle(loader)
    n_critic = 5       # D updates per G update
    lambda_gp = 10.0   # Gradient penalty weight
    
    # Main training loop
    for step in range(1, epochs+1):
        # Update discriminator multiple times
        for _ in range(n_critic):
            batch = next(data_iter).to(device)
            real_adj = densify_batch(batch, N_max=n_max)
            B = real_adj.size(0)

            # Sample latent vectors
            z = torch.randn(B, latent_dim, device=device)
            
            # Generate fake graphs
            fake_logits = G(z)
            fake_adj = torch.bernoulli(torch.sigmoid(fake_logits)).detach()

            # Discriminator scores
            d_real = D(real_adj)
            d_fake = D(fake_adj)

            # Wasserstein loss for discriminator
            loss_d = d_fake.mean() - d_real.mean()

            # Interpolate for gradient penalty
            eps = torch.rand(B, 1, 1, device=device)
            interp = eps * real_adj + (1-eps) * fake_adj
            interp.requires_grad_(True)
            
            # Gradient penalty calculation
            d_interp = D(interp)
            grad = torch.autograd.grad(
                outputs=d_interp.sum(),
                inputs=interp,
                create_graph=True
            )[0]
            gp = ((grad.view(B, -1).norm(2, dim=1) - 1)**2).mean()
            
            # Full discriminator loss
            loss_d = loss_d + lambda_gp * gp

            # Update discriminator
            opt_d.zero_grad()
            loss_d.backward()
            opt_d.step()

        # Update generator once per n_critic steps
        z = torch.randn(B, latent_dim, device=device)
        fake_logits = G(z)
        fake_adj = torch.sigmoid(fake_logits)  # Keep differentiable
        
        # Generator loss (fool discriminator)
        d_fake = D(fake_adj)
        loss_g = -d_fake.mean()  # Wasserstein generator loss

        # Update generator
        opt_g.zero_grad()
        loss_g.backward()
        opt_g.step()

        # Print progress occasionally
        if step == 1 or step % 500 == 0:
            print(f"GAN step {step:>4}/{epochs}   "
                  f"D_loss {loss_d.item():.3f}   G_loss {loss_g.item():.3f}")

    return G


@torch.no_grad()
def sample_gan_fast(G, n_graph, ref_ds, batch_size=256):
    """Vectorized batch sampling from GAN for efficiency"""
    G.eval()

    # Get node distribution
    num_nodes = np.array([d.num_nodes for d in ref_ds])
    vals, cnt = np.unique(num_nodes, return_counts=True)
    pN = cnt / cnt.sum()

    # Pre-sample all node counts
    Ns = np.random.choice(vals, size=n_graph*2, p=pN)  # Extra for rejected graphs

    out_graphs = []
    i = 0
    
    # Generate in batches until we have enough
    while len(out_graphs) < n_graph and i < len(Ns):
        b = min(batch_size, len(Ns) - i)
        batch_N = Ns[i:i+b]

        # Generate batch of latent vectors
        z = torch.randn(b, LATENT, device=device)
        logits = G(z)                               
        probs = torch.sigmoid(logits).cpu()        

        # Process each graph
        for j, N in enumerate(batch_N):
            N = int(N)  # Ensure integer
            p = probs[j, :N, :N]
            
            # Sample edges from upper triangle
            tri = (torch.rand_like(p).triu(1) < p.triu(1))
            
            # Symmetrize
            A = (tri | tri.T).int().numpy()
            
            # Create graph
            G_nx = nx.from_numpy_array(A)
            
            # Keep only connected graphs
            if nx.is_connected(G_nx):
                out_graphs.append(G_nx)
                if len(out_graphs) >= n_graph:
                    break

        i += b

    # If we still don't have enough, recursively generate more
    if len(out_graphs) < n_graph:
        more = sample_gan_fast(G, n_graph - len(out_graphs), ref_ds, batch_size)
        out_graphs.extend(more)

    return out_graphs[:n_graph]  # Trim if we got extra


# ===== EVALUATION =====

def get_statistics(graphs):
    """Extract graph statistics for evaluation"""
    stats = {
        'degree': [],      # Node degree distribution
        'clustering': [],  # Clustering coefficient
        'eigen': []        # Eigenvector centrality
    }
    
    for G in graphs:
        # Add degree values
        stats['degree'] += [d for _, d in G.degree()]
        
        # Add clustering coefficients
        stats['clustering'] += list(nx.clustering(G).values())
        
        # Add eigenvector centralities
        try:
            stats['eigen'] += list(nx.eigenvector_centrality_numpy(G).values())
        except:
            # Fall back if convergence issues
            stats['eigen'] += list(nx.degree_centrality(G).values())
    
    return stats


def novelty_and_uniqueness(gen, ref):
    """Calculate novelty (not in reference) and uniqueness (no duplicates)"""
    # Hash function for comparison
    h = lambda g: wl(g, node_attr=None, edge_attr=None)

    # Get reference hashes
    ref_hash = {h(g) for g in ref}
    seen = set()
    
    # Count metrics
    nov = uniq = both = 0
    for g in gen:
        code = h(g)
        n = code not in ref_hash
        u = code not in seen
        nov += n
        uniq += u
        both += n and u
        seen.add(code)

    # Report ratios
    N = len(gen)
    print(f"novelty {nov/N:.3f}  uniqueness {uniq/N:.3f}  both {both/N:.3f}")


# ===== MAIN EXECUTION =====
if __name__ == "__main__":
    print("\n===== Starting graph generation experiments =====\n")
    
    # Generate baseline graphs
    print("Generating baseline ER graphs...")
    baseline = sample_er_connected(1000, train_ds)

    # Train and sample from VAE
    print("\nTraining VAE model...")
    vae = train_vae(train_ld, in_dim=7, hid=256, lat=128)
    print('VAE trained!')
    
    print("Sampling from VAE...")
    vae_gen = sample_vae(vae, 1000, train_ds)
    print('VAE sampling complete!')

    # Train and sample from GAN
    print("\nTraining GAN model...")
    gan = train_gan(train_ld) 
    print('GAN trained!')
    
    print("Sampling from GAN...")
    gan_gen = sample_gan_fast(gan, 1000, train_ds)
    print('GAN sampling complete!')
   
   
    # ===== EVALUATION =====
    print("\n===== Evaluating models =====\n")
   
    # Group the different graph sets
    sets = {
        "Train": train_nx,
        "Baseline": baseline,
        "VAE": vae_gen,
        "GAN": gan_gen,
    }

    # Calculate novelty/uniqueness metrics
    for name in ["Baseline", "VAE", "GAN"]:
        print(f"\n{name}")
        novelty_and_uniqueness(sets[name], sets["Train"] + val_nx)

    # Calculate statistics for all graph sets
    stats = {k: get_statistics(v) for k, v in sets.items()}

    # Plot graph statistics
    metrics = ["degree", "clustering", "eigen"]
    fig, axes = plt.subplots(len(metrics), len(sets), figsize=(15, 9))

    for row, metric in enumerate(metrics):
        # Get range for consistent x-axis
        xmin = min(min(stats[s][metric]) for s in sets)
        xmax = max(max(stats[s][metric]) for s in sets)

        for col, (name, _) in enumerate(sets.items()):
            ax = axes[row, col]
            ax.hist(stats[name][metric], bins=20, density=True, color="#4472c4")
            ax.set_xlim(xmin, xmax)
            if col == 0:
                ax.set_ylabel(metric, fontsize=12)
            if row == 0:
                ax.set_title(name, fontsize=13)

    plt.tight_layout()
    plt.savefig("plots/graph_statistics.png", dpi=300)
    print("\nStatistics plot saved to plots/graph_statistics.png")
    
    # Sample visualization
    # Choose a random node count from training data
    node_counts = [G.number_of_nodes() for G in train_nx]
    N = random.choice(node_counts)

    # Get a graph of that size from each method
    examples = {}
    for name, graph_list in sets.items():
        candidates = [G for G in graph_list if G.number_of_nodes() == N]
        if not candidates:
            candidates = graph_list
        examples[name] = random.choice(candidates)

    # Plot example graphs
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, (name, G) in zip(axes, examples.items()):
        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos=pos, ax=ax,
                node_size=50, linewidths=0.5,
                with_labels=False,
                edge_color="#888",
                node_color="#4472c4")
        ax.set_title(f"{name}\n|V|={G.number_of_nodes()}  |E|={G.number_of_edges()}")
        ax.axis('off')

    plt.tight_layout()
    plt.savefig("plots/example_graphs.png", dpi=300)
    print("Example graphs saved to plots/example_graphs.png")
    
    print("\n===== Experiment complete =====")