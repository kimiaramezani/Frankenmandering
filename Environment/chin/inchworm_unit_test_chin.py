import networkx as nx

G_inch = nx.Graph()
G_inch.add_nodes_from(range(10))

# Edges for the pictured structure:
#  - 0,1,2 connect to 3
#  - path 3-4-5-6-7-8-9
#  - long arcs: (3,7),(3,8),(3,9),(4,8),(4,9)
edges_to_add = [
    (0,3), (1,3), (2,3),
    (3,4), (4,5), (5,6), (6,7), (7,8), (8,9),
    (3,7), (3,8), (3,9), (4,8), (4,9),
]
G_inch.add_edges_from(edges_to_add)

# Set uniform geo weights
nx.set_edge_attributes(G_inch, {e: {"weight_grid": 1.0} for e in edges_to_add})

# Opinions
opinions = {0:0, 1:0, 2:0, 3:1, 4:2, 5:3, 6:4, 7:5, 8:5, 9:5}
nx.set_node_attributes(G_inch, opinions, name="opinion")

print("G_inch nodes:", G_inch.nodes)
print("G_inch edges:", G_inch.edges)


# (Optional) quick sanity checks:
assert G_inch.number_of_nodes() == 10
assert G_inch.number_of_edges() == len(set(map(tuple, map(sorted, edges_to_add))))
for u, v, d in G_inch.edges(data=True):
    assert "weight_grid" in d

