#######################################################################
# 0. common imports & helpers  (unchanged from your last script)
#######################################################################
import os, random, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import matplotlib.pyplot as plt, networkx as nx
from tqdm import tqdm
from torch_geometric.datasets import TUDataset
from torch_geometric.loader   import DataLoader
from torch_geometric.utils    import to_networkx, to_dense_adj
from torch_geometric.nn import GCNConv, global_mean_pool
from networkx.algorithms.graph_hashing import weisfeiler_lehman_graph_hash as wl
import itertools
import torch.nn as nn


device = 'cuda' if torch.cuda.is_available() else 'cpu'
rng    = torch.Generator().manual_seed(0)
dataset= TUDataset(root='./data/', name='MUTAG')
train_ds, val_ds, test_ds = torch.utils.data.random_split(dataset,(100,44,44),
                                                          generator=rng)
train_ld = DataLoader(train_ds, batch_size=32, shuffle=True)

def graphs_from_dataset(ds):
    out=[]
    for d in ds:
        g=to_networkx(d, to_undirected=True); g.remove_edges_from(nx.selfloop_edges(g))
        out.append(g)
    return out

########################################################################
# 1.  baseline sampler  (keep yours)
########################################################################
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

########################################################################
# 2.  Graph‑level VAE  (identical to yours, shortened here)
########################################################################
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
              epochs=500):
    vae = GraphVAE(in_dim, hid, lat, N_max).to(device)
    opt = torch.optim.Adam(vae.parameters(), 1e-3)

    for ep in range(epochs):
        for batch in loader:
            batch = batch.to(device)
            adj_true = to_dense_adj(batch.edge_index,
                                    batch=batch.batch,
                                    max_num_nodes=N_max)
            adj_hat, mu, logv = vae(batch)
            recon = F.binary_cross_entropy(adj_hat, adj_true)
            kl    = 0.5 * torch.mean(mu.pow(2) + logv.exp() - 1 - logv)
            loss  = recon + kl
            opt.zero_grad(); loss.backward(); opt.step()
    return vae

@torch.no_grad()
def sample_vae(vae, num_graphs, ref_ds):
    num_nodes = np.array([d.num_nodes for d in ref_ds])
    vals, cnt = np.unique(num_nodes, return_counts=True)
    pN        = cnt / cnt.sum()

    graphs=[]
    for _ in range(num_graphs):
        N = int(np.random.choice(vals, p=pN))
        z = torch.randn(1, vae.lat_dim, device=device)
        probs = vae.decode(z, N)[0]
        tri   = (torch.rand_like(probs).triu(1) < probs.triu(1))
        A     = (tri | tri.T).int().cpu().numpy()
        graphs.append(nx.from_numpy_array(A))
    return graphs

########################################################################
# 3.  Graph‑GAN  (generator, discriminator, training, sampling)
########################################################################
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

N_MAX = max(d.num_nodes for d in dataset)        # 28 for MUTAG
LATENT = 64                                      # z‑dimension
HIDDEN = 256                                     # MLP width
LR      = 2e-4
GAN_EPOCHS = 5000                               # adversarial steps

# -------- adjacency helpers ------------------------------------------
def pyg_graph_to_dense(d, n_max=N_MAX):
    """pad a PyG *Data* object to N_max × N_max dense adjacency"""
    A = np.zeros((n_max, n_max), dtype=np.float32)
    A[d.edge_index[0], d.edge_index[1]] = 1.
    A[np.diag_indices_from(A)] = 0.
    return A

def densify_batch(batch, N_max=N_MAX):
    """
    Convert an arbitrary PyG Batch to a dense tensor  [B, N_max, N_max]
    containing 0/1 entries (self‑loops are zero).
    """
    # one call does everything:  `to_dense_adj` understands `batch.batch`
    A = to_dense_adj(batch.edge_index,
                     batch=batch.batch,
                     max_num_nodes=N_max)          # shape [B, N_max, N_max]
    return A.float()                               # BCE expects float

# -------- models ------------------------------------------------------
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

# -------- training ----------------------------------------------------

def train_gan(loader,
              epochs     = GAN_EPOCHS,   # total discriminator+generator steps
              latent_dim = LATENT,
              hidden     = HIDDEN,
              n_max      = N_MAX,
              lr         = LR):
    """
    Simple GAN training loop over `epochs` mini-batches.
    """
    # build models with the correct signature!
    G = Generator(latent_dim, hidden, n_max).to(device)
    D = Discriminator(hidden,        n_max).to(device)

    opt_g = torch.optim.Adam(G.parameters(), lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr, betas=(0.5, 0.999))
    bce   = nn.BCEWithLogitsLoss()

    ones  = lambda b: torch.ones (b, 1, device=device)
    zeros = lambda b: torch.zeros(b, 1, device=device)

    data_iter = itertools.cycle(loader)

    for step in range(1, epochs + 1):
        real_batch = next(data_iter).to(device)
        real_adj   = densify_batch(real_batch)       # [B, n_max, n_max]
        B          = real_adj.size(0)

        # ——— Update D ———————————————————————————————————
        z = torch.randn(B, latent_dim, device=device)
        fake_logits = G(z)                          # [B, n_max, n_max]
        #fake_adj    = torch.sigmoid(fake_logits).detach()
        fake_adj    = torch.bernoulli(torch.sigmoid(fake_logits))   # stochastic draw

        d_real = D(real_adj)
        d_fake = D(fake_adj)

        loss_d = 0.5 * (bce(d_real, ones(B)) + bce(d_fake, zeros(B)))
        opt_d.zero_grad(); loss_d.backward(); opt_d.step()

        # ——— Update G ———————————————————————————————————
        z = torch.randn(B, latent_dim, device=device)
        fake_logits = G(z)
        d_fake = D(torch.sigmoid(fake_logits))
        loss_g = bce(d_fake, ones(B))

        opt_g.zero_grad(); loss_g.backward(); opt_g.step()

        # ——— Logging every 500 steps ——————————————————————
        if step % 500 == 0 or step == 1:
            print(f"GAN step {step:>5}/{epochs}   "
                  f"D_loss {loss_d.item():.3f}   G_loss {loss_g.item():.3f}")

    return G


# -------- sampling ----------------------------------------------------
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
        #prob   = torch.sigmoid(logits).cpu()
        prob = torch.sigmoid(logits).cpu()
        #tri    = (torch.rand_like(prob).triu(1) < prob.triu(1))
        #A      = (tri | tri.T).int().numpy()
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
            # else: we simply drop it; we'll regenerate in a future chunk

        i += b

    # 5) If we fell short (connectivity filter), recursively top‐up
    if len(out_graphs) < n_graph:
        more = sample_gan_fast(G, n_graph - len(out_graphs), ref_ds, batch_size)
        out_graphs.extend(more)

    return out_graphs[:n_graph]


########################################################################
# 4.  statistics / novelty  (unchanged)
########################################################################
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

########################################################################
# 5.  run everything
########################################################################
train_nx = graphs_from_dataset(train_ds)
val_nx   = graphs_from_dataset(val_ds)

# --- Baseline ---------------------------------------------------------
baseline = sample_er_connected(1000, train_ds)

# --- VAE --------------------------------------------------------------
#vae     = train_vae(train_ld)
#vae     = train_vae(train_ld, in_dim=7, hid=128, lat=64)
vae     = train_vae(train_ld, in_dim=7, hid=256, lat=128)
print('VAE trained')
vae_gen = sample_vae(vae, 1000, train_ds)
print('VAE sampled')

# --- GraphRNN ---------------------------------------------------------
#gan_gen = sample_gan(train_gan(train_ld), 1000, train_ds)
gan = train_gan(train_ld)          # takes ~3–4 min on GPU
print('GAN trained')
gan_gen = sample_gan_fast(gan, 1000, train_ds)
print('GAN Fast sampled')
#gan_gen = sample_gan(gan, 1000, train_ds)
#gan_graphs = sample_gan(gan_gen, 1000, train_ds)
#print('GAN sampled')

# ------------------------------------------------------------------
# 5.  Evaluation  (replace the old loop with THIS block)
# ------------------------------------------------------------------
sets = {
    "Train"     : train_nx,
    "Baseline"  : baseline,
    "VAE"       : vae_gen,
    "GAN"       : gan_gen,
}

# ---- 1) novelty / uniqueness -------------------------------------
for name in ["Baseline", "VAE", "GAN"]:
    print(f"\n{name}")
    novelty_and_uniqueness(sets[name], sets["Train"] + val_nx)

# ---- 2) gather statistics ----------------------------------------
stats = {k: get_statistics(v) for k, v in sets.items()}

# ---- 3) single 3×4 figure ----------------------------------------
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


'''# --- Evaluation -------------------------------------------------------
for name,graphs in [("Baseline",baseline),("VAE",vae_gen),("GraphRNN",grnn_gen)]:
    print(f"\n{name}")
    novelty_and_uniqueness(graphs, train_nx+val_nx)
    st=get_statistics(graphs); tr=get_statistics(train_nx)
    plt.hist(tr['degree'], 20, alpha=.5, label='train')
    plt.hist(st['degree'],20, alpha=.5, label=name)
    plt.title(f'Degree dist – {name}'); plt.legend(); plt.show()'''
