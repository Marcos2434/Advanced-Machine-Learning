import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
import networkx as nx
from networkx.algorithms import isomorphism
from torch_geometric.utils import to_networkx


device = 'cuda'

# Load the MUTAG dataset
# Load data
dataset = TUDataset(root='./data/', name='MUTAG').to(device)
node_feature_dim = 7

# Split into training and validation
rng = torch.Generator().manual_seed(0)
train_dataset, validation_dataset, test_dataset = random_split(dataset, (100, 44, 44), generator=rng)

# Create dataloader for training and validation
train_loader = DataLoader(train_dataset, batch_size=100)
validation_loader = DataLoader(validation_dataset, batch_size=44)
test_loader = DataLoader(test_dataset, batch_size=44)


# 2.3 Sample a random graph with N nodes and edge probability r according to the Erdös-Rényi model
def sample_random_graph():
    
    # 2.1 Sample the number of nodes N from the empirical distribution of the number of nodes in the
    # training data.

    # get a list of the number of nodes in each graph from the training dataset
    num_nodes_list = [data.num_nodes for data in train_dataset]
    # get the empirical distribution of the number of nodes
    num_nodes_distribution = np.bincount(num_nodes_list)

    # normalize to make elements sum to 1 and make a probability distribution
    num_nodes_distribution = num_nodes_distribution / np.sum(num_nodes_distribution)

    # get the number of nodes N from the empirical distribution
    N = np.random.choice(np.arange(len(num_nodes_distribution)), p=num_nodes_distribution)
    # print(f"Number of nodes N: {N}")

    # Or this:
    # N = np.random.choice(num_nodes_list)
    
    # N = 11 # for testing purposes

    
    
    # 2.2.2 Compute the link probability r as the graph density (number of edges divided by total possible
    # number of edges) computed from the training graphs with N nodes.

    # Calculate the total number of edges i
    subset_of_graphs_with_N_nodes = [graph for graph in train_dataset if graph.num_nodes == N]
    num_edges_in_graphs_with_N_nodes = [graph.num_edges for graph in subset_of_graphs_with_N_nodes]

    '''
    If you see the first number of x: the number of nodes
    And then the edge_index (the second value is the number f edges)
    Then, for 23 nodes, we have 50 and 54 edges
    Data(edge_index=[2, 50], x=[23, 7], edge_attr=[50, 4], y=[1])
    Data(edge_index=[2, 54], x=[23, 7], edge_attr=[54, 4], y=[1])
    '''

    # Calculate the total possible number of edges
    possible_edges = N * (N - 1) / 2
    # Calculate the graph density for the graphs with N nodes
    graph_density = np.sum(num_edges_in_graphs_with_N_nodes) / (len(subset_of_graphs_with_N_nodes) * possible_edges)
    # print(f"Graph density: {graph_density}")
    # Calculate the link probability
    link_probability = graph_density
    # print(f"Link probability r: {link_probability}")
    
    
    
    # Create an empty adjacency matrix
    adj_matrix = np.zeros((N, N))

    # Iterate over all pairs of nodes
    for i in range(N):
        for j in range(i + 1, N):
            # Sample an edge with probability p
            if np.random.rand() < link_probability:
                adj_matrix[i, j] = 1
                adj_matrix[j, i] = 1

    return adj_matrix

# Using our function
# graph = sample_random_graph()
# print(np.all(graph == graph.T)) # check symmetry

# Modified to sample only connected graphs
def sample_connected_graph():
    while True:
        A = sample_random_graph()
        G = nx.from_numpy_array(A)
        if nx.is_connected(G):
            return A


# Using network
# N = 15
# graph_density = 0.35
# G = nx.erdos_renyi_graph(N, graph_density)
# print(G)

# from networkx.algorithms.graph_hashing import weisfeiler_lehman_graph_hash

# G1 = nx.erdos_renyi_graph(N, graph_density)
# G2 = nx.erdos_renyi_graph(N, graph_density)
# # Compute WL hashes (without using any node labels)
# h1 = weisfeiler_lehman_graph_hash(G1)
# h2 = weisfeiler_lehman_graph_hash(G2)

# print('Martin AI:')
# print(h1)
# print(h1 == h2)  # True → structure is the same


def graphs_from_dataset(dataset):
    nx_graphs = []
    for data in dataset:
        G = to_networkx(data, to_undirected=True)
        G.remove_edges_from(nx.selfloop_edges(G)) # remove self-loops if any
        nx_graphs.append(G)
    return nx_graphs


# 2.4 Novelty & Uniqueness

def novelty_and_uniqueness(graphs):
    novel = 0
    unique = 0
    novel_and_unique = 0
    seen = []

    # retrieve all training graphs in dataset
    # and convert them to networkx graphs
    train_nx = graphs_from_dataset(train_dataset)

    for G in tqdm(graphs):
        is_novel  = not any(nx.is_isomorphic(G, H) for H in train_nx) # Novel if not iso to any training graph
        is_unique = not any(nx.is_isomorphic(G, H) for H in seen) # Unique if not iso to any previously seen sample
        
        if is_novel:  novel += 1
        if is_unique: unique += 1
        if is_novel and is_unique: novel_and_unique += 1

        seen.append(G)

    print(f"Novelty: {novel/M:.3f}")
    print(f"Uniqueness: {unique/M:.3f}")
    print(f"Novel & Unique: {novel_and_unique/M:.3f}")



# example

# sample M random graphs
M = 1000
samples = [
    nx.from_numpy_array(sample_connected_graph())
    for _ in range(M)
]

novelty_and_uniqueness(samples)


# Provide a graph level VAE implementation
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, BatchNorm
from torch_geometric.utils import to_dense_adj, to_dense_batch

class GraphLevelVAE(torch.nn.Module):
    def __init__(self, in_dim, hid_dim, lat_dim, dropout=0.5):
        super().__init__()
        
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.lat_dim = lat_dim
        self.lambda_conn = 5000  # penalize isolated nodes
        
        # encoder GNN
        self.conv1 = GCNConv(in_dim, hid_dim)
        self.bn1   = BatchNorm(hid_dim)
        self.conv2 = GCNConv(hid_dim, hid_dim)
        self.bn2   = BatchNorm(hid_dim)
        self.conv3 = GCNConv(hid_dim, hid_dim)
        self.bn3   = BatchNorm(hid_dim)
        self.dropout = dropout
        
        self.node_proj = nn.Linear(in_dim, hid_dim)
        
        # decoder MLP
        self.dec = nn.Sequential(
            nn.Linear(lat_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim),
        )
        
        self.edge_decoder = nn.Sequential(
            nn.Linear(2*hid_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, 1),
        )


        # balance edge vs non-edge in BCE
        densities = [
            G.number_of_edges() /
            (G.number_of_nodes()*(G.number_of_nodes()-1)/2)
            for G in graphs_from_dataset(train_dataset)
        ]
        p_pos      = np.mean(densities)
        pos_weight = torch.tensor((1-p_pos)/p_pos, device=device)
        self.bce   = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        # initialize edge_decoder bias so logits start near logit(p_pos)
        init_bias = torch.log(pos_weight) * -1  # since pos_weight=(1−p)/p ⇒ logit(p)=−log(pos_weight)
        self.edge_decoder[-1].bias.data.fill_(init_bias)

        
        # Graph-level latent parameters
        self.lin_mu     = torch.nn.Linear(hid_dim, lat_dim)
        self.lin_logvar = torch.nn.Linear(hid_dim, lat_dim)
    
    def encode(self, x, edge_index, batch):
        x = F.leaky_relu(self.bn1(self.conv1(x, edge_index)), 0.2)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.leaky_relu(self.bn2(self.conv2(x, edge_index)), 0.2)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.leaky_relu(self.bn3(self.conv3(x, edge_index)), 0.2)
        g = global_mean_pool(x, batch)
        return self.lin_mu(g), self.lin_logvar(g)
    
    # def encode(self, x, edge_index, batch):
    #     """
    #     We first compute node embeddings via two rounds of graph convolution,
    #     then squash all nodes in each graph down to a single vector by averaging. 
    #     That single vector g is your graph-level representation.
    #     """
    #     x = F.relu(self.conv1(x, edge_index))
    #     x = F.relu(self.conv2(x, edge_index))
    #     x = F.relu(self.conv3(x, edge_index))
    #     g = global_mean_pool(x, batch) # For each graph i, take the mean of all node‐embeddings x[j] where batch[j]==i.
        
    #     # turn (and return) pooled graph vector g into mean and logvar
    #     # into the parameters of a multivariate Gaussian in latent space
    #     # this output is the mean and logvar of the latent variable z
    #     # in our approximate posterior q(z|x)
    #     return self.lin_mu(g), self.lin_logvar(g)
    
    def decode(self, z, x, batch):
        # 1) per‑graph component from latent
        h_graph = self.dec(z)                             # [B, H]

        # 2) per‑node component from the encoder's raw node features
        x_dense, mask = to_dense_batch(x, batch)          # [B, N_max, in_dim]
        h_nodes = self.node_proj(x_dense)                 # [B, N_max, H]

        # 3) inject the graph‑level info into every node
        h_nodes = h_nodes + h_graph.unsqueeze(1)          # broadcast add

        # 4) edge scores exactly as before
        B, N, H = h_nodes.size()
        h1 = h_nodes.unsqueeze(2).expand(B, N, N, H)
        h2 = h_nodes.unsqueeze(1).expand(B, N, N, H)
        pair   = torch.cat([h1, h2], dim=-1)              # [B,N,N,2H]
        logits = self.edge_decoder(pair).squeeze(-1)      # [B,N,N]
        return logits, mask
    
    # def decode(self, z, x, batch):
    #     h = self.dec(z)                              # [B, H]
    #     x0 = x[:, :1]; _, mask = to_dense_batch(x0, batch)
    #     N = mask.size(1)
    #     h_nodes = h.unsqueeze(1).expand(-1, N, -1)   # [B,N,H]

    #     B, N, H = h_nodes.size()
    #     h1 = h_nodes.unsqueeze(2).expand(B, N, N, H)
    #     h2 = h_nodes.unsqueeze(1).expand(B, N, N, H)
    #     pair   = torch.cat([h1, h2], dim=-1)         # [B,N,N,2H]
    #     logits = self.edge_decoder(pair).squeeze(-1) # [B,N,N]
    #     return logits, mask                          # raw logits
    
    # def decode(self, z, x, batch):
    #     h = self.dec(z)
        
        
    #     x0 = x[:, :1]                           # [num_nodes, 1]
    #     _, mask = to_dense_batch(x0, batch)    # mask: [batch, N_max]
    #     N_max = mask.size(1)
        
    #     h_nodes = h.unsqueeze(1).expand(-1, N_max, -1)
        
    #     # adjacency logits are computed via inner-product
    #     adj_logits = torch.matmul(h_nodes, h_nodes.transpose(-1, -2))
        
    #     return torch.sigmoid(adj_logits), mask     # [b, N_max, N_max]
    
    def reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)
    
    @torch.no_grad()
    def sample_from_vae(self, num_samples=5):
        self.eval()
        device = next(self.parameters()).device
        samples = []
        for _ in range(num_samples):
            # 1) sample z ~ N(0,I)
            z = torch.randn(1, self.lat_dim, device=device)
            # 2) pass through your decoder MLP
            h = self.dec(z)                              # [1, H]
            # 3) sample N nodes
            num_nodes_list = [d.num_nodes for d in train_dataset]
            node_dist      = np.bincount(num_nodes_list)
            node_dist     = node_dist / node_dist.sum()
            N = np.random.choice(len(node_dist), p=node_dist)
            # 4) tile to per-node
            h_nodes = h.unsqueeze(1).expand(1, N, -1)    # [1, N, H]
            B, N, H = h_nodes.shape
            # 5) build every (i,j) pair and score
            h1 = h_nodes.unsqueeze(2).expand(B, N, N, H)
            h2 = h_nodes.unsqueeze(1).expand(B, N, N, H)
            pair = torch.cat([h1, h2], dim=-1)           # [1,N,N,2H]
            logits = self.edge_decoder(pair).squeeze(-1)[0]   # [N,N]
            probs  = torch.sigmoid(logits)
            # 6) sample edges
            tri = torch.triu(torch.rand_like(probs), diagonal=1) < torch.triu(probs, diagonal=1)
            A = tri.int().cpu().numpy()
            A = A + A.T
            samples.append(nx.from_numpy_array(A))
        return samples

    def sample_connected_graph(self, num_samples=1):
        samples = []
        for _ in range(num_samples):
            while True:
                G = self.sample_from_vae(1)[0]
                if nx.is_connected(G):
                    samples.append(G)
                    break
        return samples

    def forward(self, data):
        mu, logvar     = self.encode(data.x, data.edge_index, data.batch)
        z              = self.reparam(mu, logvar)
        logits, mask   = self.decode(z, data.x, data.batch)
        adj_gt         = to_dense_adj(data.edge_index, data.batch)
        return logits, adj_gt, mask, mu, logvar
    
    # def forward(self, data):
    #     # forward pass
    #     mu, logvar = self.encode(data.x, data.edge_index, data.batch)
        
    #     # encoder
    #     mu, logvar = self.encode(data.x, data.edge_index, data.batch)
        
    #     # reparameterization trick
    #     z = self.reparam(mu, logvar)

    #     # decoder
    #     adj_pred, mask = self.decode(z, data.x, data.batch)
    #     adj_gt = to_dense_adj(data.edge_index, data.batch)  # [b, N_max, N_max]
        
    #     # forward returns the predicted adjacency matrix, the ground truth adjacency matrix, the mask, and the latent variables 
    #     # because we need them for the loss function
    #     return adj_pred, adj_gt, mask, mu, logvar
        
model = GraphLevelVAE(in_dim=node_feature_dim, hid_dim=128, lat_dim=64, dropout=0.2)
model = model.to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

# load model if exists
# train = False
# try:
#     model.load_state_dict(torch.load("models/vae.pth"))
#     print("Model loaded")
# except FileNotFoundError:
#     print("Model not found, starting training")
#     train = True
    
train = True
kl_anneal_epochs = 200

if train:
    epochs = 1000
    for epoch in (pbar := tqdm(range(epochs))):
        model.train()
        
        kl_weight = min(1.0, epoch/kl_anneal_epochs)
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            logits, adj_gt, mask, mu, logvar = model(batch)
            

            # weighted BCE on raw logits
            loss_recon = model.bce(logits[mask], adj_gt[mask])
            loss_kl    = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            

            probs = torch.sigmoid(logits)            # [B, N, N]
            deg   = probs.sum(-1)                    # expected degree per node
            iso_pen = F.relu(1.0 - deg)              # >0 only for “nearly isolated” nodes
            conn_loss = (iso_pen * mask.float()).sum() / mask.sum()

            loss = loss_recon + kl_weight*loss_kl + model.lambda_conn*conn_loss

            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * batch.num_graphs
            
            
            
        pbar.set_description(f"Loss: {total_loss/len(train_dataset):.4f}")

torch.save(model.state_dict(), "models/vae.pth")


dgm_samples = model.sample_connected_graph(num_samples=3)
# Display graphs
for i, G in enumerate(dgm_samples):
    print(f"Sample {i}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    nx.draw(G, node_size=50); plt.show()

# 2.5 Statistics
def get_statistics(graphs):
    stats = {
        "degree":      [],
        "clustering":  [],
        "eigenvector": []
    }
    for G in graphs:
        # degrees of all nodes
        stats["degree"].extend([d for _, d in G.degree()]) # G.degree() returns a tuple (node, degree)
        
        # local clustering coefficients
        stats["clustering"].extend(list(nx.clustering(G).values()))
        
        # eigenvector centrality (may need convergence tweaks on large graphs)
        stats["eigenvector"].extend(list(nx.eigenvector_centrality_numpy(G).values()))
    return stats


# retrieve all training graphs in dataset
# and convert them to networkx graphs
train_nx = graphs_from_dataset(train_dataset)

train_stats  = get_statistics(train_nx)
sample_stats = get_statistics(samples) # baseline
dgm_stats = get_statistics(dgm_samples) # deep generative model

# plotting
fig, axes = plt.subplots(3, 3, figsize=(12, 10))
metrics = ["degree", "clustering", "eigenvector"]
for i, m in enumerate(metrics):
    
    axes[i, 0].hist(train_stats[m], bins=20, density=True)
    axes[i, 0].set_title(f"Training dataset {m}")

    axes[i, 1].hist(sample_stats[m], bins=20, density=True)
    axes[i, 1].set_title(f"Baseline {m}")

    axes[i, 2].hist(dgm_stats[m], bins=20, density=True)
    axes[i, 2].set_title(f"Deep Generative Model {m}")
    
    
plt.tight_layout()
plt.savefig("stats.png")
plt.show()


print("Novelty & Uniqueness for baseline samples")
novelty_and_uniqueness(samples)

print("Novelty & Uniqueness for deep generative model samples")
novelty_and_uniqueness(dgm_samples)