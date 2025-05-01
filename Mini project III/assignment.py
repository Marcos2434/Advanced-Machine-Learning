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


device = 'mps'

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
M = 100
samples = [
    nx.from_numpy_array(sample_connected_graph())
    for _ in range(M)
]

novelty_and_uniqueness(samples)


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

# usage example:

# retrieve all training graphs in dataset
# and convert them to networkx graphs
train_nx = graphs_from_dataset(train_dataset)

train_stats  = get_statistics(train_nx)
sample_stats = get_statistics(samples)

# plotting
fig, axes = plt.subplots(3, 2, figsize=(8, 10))
metrics = ["degree", "clustering", "eigenvector"]
for i, m in enumerate(metrics):
    axes[i,0].hist(train_stats[m],  bins=20, density=True)
    axes[i,0].set_title(f"Train {m}")
    axes[i,1].hist(sample_stats[m], bins=20, density=True)
    axes[i,1].set_title(f"Sampled {m}")
plt.tight_layout()
plt.savefig("stats.png")
# plt.show()

# Provide a graph level VAE implementation



import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.utils import to_dense_adj, to_dense_batch


class GraphLevelVAE(torch.nn.Module):
    def __init__(self, in_dim, hid_dim, lat_dim):
        super().__init__()
        
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.lat_dim = lat_dim
    
    def encoder():
        pass
    
    def decoder():
        pass
    
    def forward(self, data):
        # forward pass
        mu, logvar = self.encode(data.x, data.edge_index, data.batch)
        print(data.x)
        print(data.batch)
        
        # encoder
        # decoder
        
model = GraphLevelVAE(in_dim=node_feature_dim, hid_dim=64, lat_dim=32).to(device)
# opt = torch.optim.Adam(model.parameters(), lr=1e-3)

epochs = 10
for epoch in tqdm(range(epochs)):
    model.train()
    
    total_loss = 0.0
    for batch in train_loader:
        batch = batch.to(device)
        
        # adj_pred is the predicted adjacency matrix with probabilities instead of entries
        # adj_gt is the ground truth adjacency matrix
        # mask is a boolean tensor indicating which elements to compute the loss for
        # mu and logvar are the latent variables
        adj_pred, adj_gt, mask, mu, logvar = model(batch) 
        
        
        # only compute BCE where mask==True
        loss_recon = F.binary_cross_entropy(
            adj_pred[mask], adj_gt[mask]
        )
        
        loss_kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = loss_recon + loss_kl
        # opt.zero_grad(); loss.backward(); opt.step()
        total_loss += loss.item() * batch.num_graphs
        
    print(f"Epoch {epoch:02d} | Loss: {total_loss/len(train_dataset):.4f}")


# class GraphLevelVAE(torch.nn.Module):
#     def __init__(self, in_dim, hid_dim, lat_dim):
#         super().__init__()
#         # Encoder GNN
#         self.conv1 = GCNConv(in_dim, hid_dim)
#         self.conv2 = GCNConv(hid_dim, hid_dim)
#         # Graph-level latent parameters
#         self.lin_mu     = torch.nn.Linear(hid_dim, lat_dim)
#         self.lin_logvar = torch.nn.Linear(hid_dim, lat_dim)
#         # Decoder MLP
#         self.dec1 = torch.nn.Linear(lat_dim, hid_dim)
#         self.dec2 = torch.nn.Linear(hid_dim, hid_dim)

#     def encode(self, x, edge_index, batch):
#         x = F.relu(self.conv1(x, edge_index))
#         x = F.relu(self.conv2(x, edge_index))
#         g = global_mean_pool(x, batch)             # [b, hid_dim]
#         return self.lin_mu(g), self.lin_logvar(g)

#     def reparam(self, mu, logvar):
#         std = torch.exp(0.5 * logvar)
#         return mu + std * torch.randn_like(std)

#     def decode(self, z, batch):
#         h = F.relu(self.dec1(z))
#         h = F.relu(self.dec2(h))                   # [b, hid_dim]
#         # expand per-graph z to per-node embeddings
#         h_nodes, mask = to_dense_batch(h, batch)   # [b, N_max, hid_dim]
#         # adjacency logits via inner-product
#         adj_logits = torch.matmul(h_nodes, h_nodes.transpose(-1,-2))
#         return torch.sigmoid(adj_logits), mask     # [b, N_max, N_max]

#     def forward(self, data):
#         mu, logvar = self.encode(data.x, data.edge_index, data.batch)
#         z = self.reparam(mu, logvar)

#         adj_pred, mask = self.decode(z, data.batch)
#         adj_gt = to_dense_adj(data.edge_index, data.batch)  # [b, N_max, N_max]
#         return adj_pred, adj_gt, mask, mu, logvar


# model = GraphLevelVAE(in_dim=node_feature_dim, hid_dim=64, lat_dim=32).to(device)
# opt   = torch.optim.Adam(model.parameters(), lr=1e-3)

# for epoch in tqdm(range(1, 51)):
#     model.train()
#     total_loss = 0
#     for batch in train_loader:
#         batch = batch.to(device)
#         adj_p, adj_t, mask, mu, logvar = model(batch)
#         # only compute BCE where mask==True
#         loss_recon = F.binary_cross_entropy(
#             adj_p[mask], adj_t[mask]
#         )
#         loss_kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
#         loss = loss_recon + loss_kl
#         opt.zero_grad(); loss.backward(); opt.step()
#         total_loss += loss.item() * batch.num_graphs
#     print(f"Epoch {epoch:02d} | Loss: {total_loss/len(train_dataset):.4f}")