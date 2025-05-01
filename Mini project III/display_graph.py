import random
import matplotlib.pyplot as plt
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx
import networkx as nx

# Load (or reuse your existing) MUTAG dataset
dataset = TUDataset(root='./data/', name='MUTAG')

# Pick a random index
idx = random.randrange(len(dataset))
data = dataset[idx]

# Convert to NetworkX
G = to_networkx(data, to_undirected=True)
G.remove_edges_from(nx.selfloop_edges(G))

# Draw it
plt.figure(figsize=(6,6))
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=False, node_size=50, edge_color='gray')
plt.title(f"MUTAG sample #{idx} —  nodes: {G.number_of_nodes()}, edges: {G.number_of_edges()}")
plt.show()