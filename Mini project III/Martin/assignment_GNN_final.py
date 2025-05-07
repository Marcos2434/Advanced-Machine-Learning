# Mini Project 3
import os, random, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import matplotlib.pyplot as plt, networkx as nx
from tqdm import tqdm
from torch_geometric.datasets import TUDataset
from torch_geometric.loader   import DataLoader
from torch_geometric.utils    import to_networkx, to_dense_adj
from torch_geometric.nn import GCNConv, global_mean_pool
from networkx.algorithms.graph_hashing import weisfeiler_lehman_graph_hash as wl
import itertools
import torch
import torch.nn as nn
import random

########################
### CONSTANTS ###

def graphs_from_dataset(ds):
    out=[]
    for d in ds:
        g=to_networkx(d, to_undirected=True); g.remove_edges_from(nx.selfloop_edges(g))
        out.append(g)
    return out

device = 'cuda' if torch.cuda.is_available() else 'cpu'
rng    = torch.Generator().manual_seed(0)
dataset= TUDataset(root='./data/', name='MUTAG')
train_ds, val_ds, test_ds = torch.utils.data.random_split(dataset,(100,44,44),
                                                          generator=rng)
train_ld = DataLoader(train_ds, batch_size=32, shuffle=True)

train_nx = graphs_from_dataset(train_ds)
val_nx   = graphs_from_dataset(val_ds)

all_degs = [d for G in train_nx for _, d in G.degree()]
MAX_DEG = float(np.percentile(all_degs, 95))
N_MAX = max(d.num_nodes for d in dataset)        # 28 for MUTAG
LATENT = 128                                      # z‑dimension
HIDDEN = 256                                     # MLP width
LR      = 1e-4
GAN_EPOCHS = 1000                               # adversarial steps
## --------------------------------------



##### BASELINE #####
def sample_er_connected(n_graph, ref_ds):
    num_nodes = np.array([d.num_nodes for d in ref_ds])
    vals,cnt  = np.unique(num_nodes, return_counts=True)
    pN        = cnt/cnt.sum()

    outs=[]
    for _ in range(n_graph):
        N=int(np.random.choice(vals,p=pN))
        edges=[d.num_edges for d in ref_ds if d.num_nodes==N]
        dens = np.mean([e/(N*(N-1)/2) for e in edges])
        while True:
            g = nx.erdos_renyi_graph(N, dens)
            if nx.is_connected(g): break
        outs.append(g)
    return outs


##### 2.  Graph‑VAE #####

class GraphVAE(nn.Module):
    """
    • Encoder : 2‑layer GCN → mean‑pool → μ, logσ² (R^lat_dim)
    • Decoder : MLP that outputs a dense N×N adjacency for the
                *largest* N seen in the batch.
      At sample time we simply slice [:N,:N].
    """
    def __init__(self, in_dim, hid_dim, lat_dim, N_max):
        super().__init__()
        self.N_max = N_max
        self.lat_dim = lat_dim

        # --- encoder ---
        self.enc1 = GCNConv(in_dim, hid_dim)
        self.enc2 = GCNConv(hid_dim, hid_dim)
        self.mu    = nn.Linear(hid_dim, lat_dim)
        self.logv  = nn.Linear(hid_dim, lat_dim)

        # --- decoder ---
        self.dec = nn.Sequential(
            nn.Linear(lat_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, N_max * N_max)   # <<–– produces N_max² logits
        )

    def encode(self, x, edge_index, batch):
        h = F.relu(self.enc1(x, edge_index))
        h = F.relu(self.enc2(h, edge_index))
        g = global_mean_pool(h, batch)
        mu, logv = self.mu(g), self.logv(g)
        std = torch.exp(0.5 * logv)
        z   = mu + std * torch.randn_like(std)
        return z, mu, logv

    def decode(self, z, N):
        """returns a symmetric (N×N) edge‑probability matrix"""
        B = z.size(0)
        logits = self.dec(z).view(B, self.N_max, self.N_max)
        logits = (logits + logits.transpose(1, 2)) / 2
        logits = logits.masked_fill(
            torch.eye(self.N_max, device=z.device).bool(), -9e15
        )
        probs  = torch.sigmoid(logits)           # [B, N_max, N_max]
        return probs[:, :N, :N]                  # slice to real size

    def forward(self, data):
        z, mu, logv = self.encode(data.x, data.edge_index, data.batch)
        N = data.num_nodes                       # longest graph in batch
        adj_hat = self.decode(z, N)
        return adj_hat, mu, logv

def train_vae(loader, in_dim=7, hid=128, lat=64,
              N_max=max(d.num_nodes for d in dataset),
              epochs=5000):
    vae = GraphVAE(in_dim, hid, lat, N_max).to(device)
    opt = torch.optim.Adam(vae.parameters(), 1e-3)
    for ep in range(epochs):
        for batch in loader:
            batch = batch.to(device)
            adj_true = to_dense_adj(batch.edge_index,
                                    batch=batch.batch,
                                    max_num_nodes=N_max)
            adj_hat, mu, logv = vae(batch)
            loss_recon = F.binary_cross_entropy(adj_hat, adj_true)
            loss_kl    = 0.5 * torch.mean(mu.pow(2) + logv.exp() - 1 - logv)

            loss =  (loss_recon + 0.6 * loss_kl)        # KL annealed to 0.6
            opt.zero_grad(); loss.backward(); opt.step()
        
        if ep == 1 or ep % 200 == 0:
            print(f"VAE epoch {ep:>4}/{epochs}   "
                  f"loss {loss.item():.3f} ")
    return vae

@torch.no_grad()
def sample_vae(vae, num_graphs, ref_ds):
    """Same as sample_vae but keeps only connected graphs."""
    # empirical node-count distribution
    num_nodes = np.array([d.num_nodes for d in ref_ds])
    vals, cnt  = np.unique(num_nodes, return_counts=True)
    pN         = cnt / cnt.sum()

    out = []
    while len(out) < num_graphs:       # ← use num_graphs here
        N = int(np.random.choice(vals, p=pN))
        z = torch.randn(1, vae.lat_dim, device=device)
        probs = vae.decode(z, N)[0]              # [N,N] edge-probs
        tri   = (torch.rand_like(probs).triu(1) < probs.triu(1))
        A     = (tri | tri.T).int().cpu().numpy()
        G     = nx.from_numpy_array(A)
        if nx.is_connected(G):
            out.append(G)
    return out


##### 3.  Graph‑GAN  #####
# (generator, discriminator, training, sampling)
#
#  – A *fixed‑size* representation is used just like in the VAE:
#    the generator always outputs an N_max × N_max adjacency matrix
#    (zeros on the diagonal, symmetric).  At sample‑time we slice
#    [:N, :N] where N is drawn from the empirical node‑count
#    distribution of the training set.
#
#  – Because MUTAG graphs are tiny (≤ 28 nodes) a simple MLP
#    suffices; no GCN is required.
#
# ---------------------------------------------------------------------


# -------- adjacency helpers ------------------------------------------
def pyg_graph_to_dense(d, n_max=N_MAX):
    """pad a PyG *Data* object to N_max × N_max dense adjacency"""
    A = np.zeros((n_max, n_max), dtype=np.float32)
    A[d.edge_index[0], d.edge_index[1]] = 1.
    A[np.diag_indices_from(A)] = 0.
    return A

# Models
class Generator(nn.Module):
    def __init__(self, z_dim=LATENT, hidden=HIDDEN, n_max=N_MAX):
        super().__init__()
        out_len = n_max * n_max
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden, out_len),
        )
        self.n_max = n_max

    def forward(self, z):
        logits = self.net(z)                      # [B, N_max²]
        logits = logits.view(-1, self.n_max, self.n_max)
        logits = (logits + logits.transpose(1, 2)) / 2        # sym.
        logits = logits.masked_fill(
            torch.eye(self.n_max, device=logits.device).bool(), -9e15
        )
        return logits      # we keep *logits* – BCEWithLogitsLoss later

class Discriminator(nn.Module):
    def __init__(self, hidden=HIDDEN, n_max=N_MAX):
        super().__init__()
        inp_len = n_max * n_max
        self.net = nn.Sequential(
            nn.Linear(inp_len, hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden, 1)
        )

    def forward(self, adj):
        flat = adj.view(adj.size(0), -1)          # [B, N_max²]
        return self.net(flat)                     # logits real/fake

# Training
def densify_batch(batch, N_max=N_MAX):
    # turns a PyG Batch into a [B, N_max, N_max] float tensor of 0/1 edges
    return to_dense_adj(batch.edge_index, batch=batch.batch,
                        max_num_nodes=N_max).float()

def train_gan(loader,
              epochs     = GAN_EPOCHS,   # total discriminator+generator steps
              latent_dim = LATENT,
              hidden_dim = HIDDEN,
              n_max      = N_MAX,
              lr         = LR):
    """
    WGAN-GP on adjacency matrices.
    """

    # 1) instantiate models
    G = Generator(z_dim=latent_dim, hidden=hidden_dim, n_max=n_max).to(device)
    D = Discriminator(hidden=hidden_dim,        n_max=n_max).to(device)

    opt_g = torch.optim.Adam(G.parameters(), lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr, betas=(0.5, 0.999))

    # non-saturating GAN loss for WGAN-GP
    one  = lambda B: torch.ones (B, 1, device=device)
    zero = lambda B: torch.zeros(B, 1, device=device)

    data_iter = itertools.cycle(loader)
    n_critic  = 5    # #D updates per G update
    lambda_gp      = 10.0 # gradient-penalty weight
    for step in range(1, epochs+1):
        # ——— 1) update D n_critic times —————————————————————————
        for _ in range(n_critic):
            batch     = next(data_iter).to(device)
            real_adj  = densify_batch(batch, N_max=n_max)  # [B, n_max, n_max]
            B         = real_adj.size(0)

            # sample fake
            z         = torch.randn(B, latent_dim, device=device)
            fake_logits = G(z)                             # [B, n_max, n_max]
            # stochastically binarize for the critic:
            fake_adj    = torch.bernoulli(torch.sigmoid(fake_logits)).detach()

            # discriminator scores
            d_real = D(real_adj)   # [B,1]
            d_fake = D(fake_adj)   # [B,1]

            # WGAN loss
            loss_d = d_fake.mean() - d_real.mean()

            # gradient penalty
            eps   = torch.rand(B, 1, 1, device=device)      # <-- note shape (B,1,1)
            interp= eps * real_adj + (1-eps) * fake_adj
            interp.requires_grad_(True)
            d_interp = D(interp)
            # compute gradient D(interp) w.r.t. interp
            grad = torch.autograd.grad(
                outputs=d_interp.sum(),      # sum over batch
                inputs=interp,
                create_graph=True
            )[0]
            gp = ((grad.view(B, -1).norm(2, dim=1) - 1)**2).mean()
            loss_d = loss_d + lambda_gp * gp

            opt_d.zero_grad()
            loss_d.backward()
            opt_d.step()

        # ——— 2) update G once —————————————————————————————————
        z       = torch.randn(B, latent_dim, device=device)
        fake_logits = G(z)
        fake_adj    = torch.sigmoid(fake_logits)    # \hat A_b (soft). keep it differentiable
        d_fake = D(fake_adj)                        # f_\theta(\hat A_b)
        loss_g = -d_fake.mean()                     # minus sign

        opt_g.zero_grad()
        loss_g.backward()
        opt_g.step()

        # logging
        if step == 1 or step % 500 == 0:
            print(f"GAN step {step:>4}/{epochs}   "
                  f"D_loss {loss_d.item():.3f}   G_loss {loss_g.item():.3f}")

    return G


##### SAMPLING #####

@torch.no_grad()
def sample_gan(G, n_graph, ref_ds, N_max=N_MAX):
    """
    Draw n_graph graphs from the trained generator *G*.
    We sample the node count N from the empirical distribution found
    in `ref_ds`, then slice the N×N top‑left corner of the generated
    logits and threshold at 0.5.
    """
    # empirical node–count distribution
    num_nodes = np.array([d.num_nodes for d in ref_ds])
    vals, cnt = np.unique(num_nodes, return_counts=True)
    pN        = cnt / cnt.sum()

    G.eval(); graphs = []
    while len(graphs) < n_graph:
        N = int(np.random.choice(vals, p=pN))
        z = torch.randn(1, LATENT, device=device)
        logits = G(z)[0, :N, :N]                   # slice to true size
        prob = torch.sigmoid(logits).cpu()
        A    = torch.bernoulli(prob).int().numpy()
        A    = np.triu(A,1); A = A + A.T       # keep symmetry
        g      = nx.from_numpy_array(A)
        if nx.is_connected(g): graphs.append(g)
    return graphs

# Fast sampling
@torch.no_grad()
def sample_gan_fast(G, n_graph, ref_ds, batch_size=256):
    """
    Vectorized GAN sampling:
     - we first pick all `n_graph` node‐counts N_i,
     - then do one or more big forward‐passes through G,
     - then slice out each N_i×N_i probability map,
     - threshold, and connectivity‐filter.
    """
    G.eval()

    # 1) Empirical distribution of node counts
    num_nodes = np.array([d.num_nodes for d in ref_ds])
    vals, cnt  = np.unique(num_nodes, return_counts=True)
    pN         = cnt / cnt.sum()

    # 2) Pre‐sample all N_i at once
    Ns = np.random.choice(vals, size=n_graph, p=pN)

    out_graphs = []
    i = 0
    # We'll generate in chunks of up to batch_size
    while i < n_graph:
        b = min(batch_size, n_graph - i)
        batch_N = Ns[i : i + b]

        # 3) Draw one big batch of z's and forward
        z = torch.randn(b, LATENT, device=device)
        logits = G(z)                               # [b, N_MAX, N_MAX]
        probs  = torch.sigmoid(logits).cpu()        # move to CPU once

        # 4) Slice & threshold each slice
        for j, N in enumerate(batch_N):
            p = probs[j, :N, :N]
            tri = (torch.rand_like(p).triu(1) < p.triu(1))
            A   = (tri | tri.T).int().numpy()
            G_nx = nx.from_numpy_array(A)
            if nx.is_connected(G_nx):
                out_graphs.append(G_nx)

        i += b

    if len(out_graphs) < n_graph: # recursively top‐up
        more = sample_gan_fast(G, n_graph - len(out_graphs), ref_ds, batch_size)
        out_graphs.extend(more)

    return out_graphs[:n_graph]


##### STATISTICS  #####

def get_statistics(graphs):
    stats={'degree':[],'clustering':[],'eigen':[]}
    for G in graphs:
        stats['degree']     += [d for _,d in G.degree()]
        stats['clustering'] += list(nx.clustering(G).values())
        stats['eigen']      += list(nx.eigenvector_centrality_numpy(G).values())
    return stats

def novelty_and_uniqueness(gen, ref):
    # WL hash without considering node labels or edge labels
    h = lambda g: wl(g, node_attr=None, edge_attr=None)

    ref_hash = {h(g) for g in ref}
    seen = set()
    nov = uniq = both = 0
    for g in gen:
        code = h(g)
        n = code not in ref_hash
        u = code not in seen
        nov += n
        uniq += u
        both += n and u
        seen.add(code)

    N = len(gen)
    print(f"novelty {nov/N:.3f}  uniqueness {uniq/N:.3f}  both {both/N:.3f}")



##### MAIN PART #####
##############################

# Baseline
baseline = sample_er_connected(1000, train_ds)

# VAE
vae     = train_vae(train_ld, in_dim=7, hid=256, lat=128)
print('VAE trained')
vae_gen = sample_vae(vae, 1000, train_ds)
print('VAE sampled')
print("share connected:", sum(nx.is_connected(g) for g in vae_gen)/3.)
# GAN
gan = train_gan(train_ld) 
print('GAN trained')
gan_gen = sample_gan_fast(gan, 1000, train_ds)
print('GAN Fast sampled')


##### EVALUATION METRICS #####
##############################

sets = {
    "Train"     : train_nx,
    "Baseline"  : baseline,
    "VAE"       : vae_gen,
    "GAN"       : gan_gen,
}

# Novelty / uniqueness
for name in ["Baseline", "VAE", "GAN"]:
    print(f"\n{name}")
    novelty_and_uniqueness(sets[name], sets["Train"] + val_nx)

# Gather statistics
stats = {k: get_statistics(v) for k, v in sets.items()}

# Plotting
metrics = ["degree", "clustering", "eigen"]       # three rows
fig, axes = plt.subplots(len(metrics), len(sets), figsize=(15, 9))

for row, metric in enumerate(metrics):
    xmin = min(min(stats[s][metric]) for s in sets)  # same range per row
    xmax = max(max(stats[s][metric]) for s in sets)

    for col, (name, _) in enumerate(sets.items()):
        ax   = axes[row, col]
        ax.hist(stats[name][metric], bins=20, density=True, color="#4472c4")
        ax.set_xlim(xmin, xmax)
        if col == 0:
            ax.set_ylabel(metric, fontsize=12)
        if row == 0:
            ax.set_title(name, fontsize=13)

plt.tight_layout()
plt.show()

# Sampling

# Pick a node number so that they all sample the same graph size
node_counts = [G.number_of_nodes() for G in train_nx]
N = random.choice(node_counts)

# Grab one graph
examples = {}
for name, graph_list in sets.items():
    candidates = [G for G in graph_list if G.number_of_nodes() == N] # filter by size
    if len(candidates) == 0:
        candidates = graph_list
    examples[name] = random.choice(candidates)

# Plotting
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
plt.show()