import networkx as nx

from torch_geometric.utils import is_undirected, to_undirected

# Create an undirected graph
G = nx.Graph()
G.add_edge(0, 1)
G.add_edge(1, 2)

print("Original Graph Undirected or Directed:", is_undirected(G, num_nodes=3))  # should be True
# Create a directed graph using the constructor
H = nx.DiGraph(G)

# Inspect the edges of the directed graph
print("DiGraph:", list(H.edges()))

I = G.to_directed()
print("to_directed:", list(I.edges()))