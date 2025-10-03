import numpy as np
import torch
from torch_geometric.data import Data, HeteroData
import gymnasium as gym
from gymnasium import spaces
from gerry_environment_chin import FrankenData, FrankenmanderingEnv
from init_graph_to_frankendata import graph_to_frankendata
from make_grid_chin import Graph
from torch_geometric.data import Data, HeteroData

# ---- Inchworm test ----
# Convert to FrankenData (also attaches a full HeteroData at fd.hetero)
# We will later write a generator function which yields different FrankenData objects for different districts 
# and number of voters (Nodes).

K = 6   # num districts
N = 72  # num nodes
# Build graph
G = Graph(*Graph.make_node_ids(N))
G.generate_positions(mode="grid", H=8, W=9)
G.build_edges_grid(H=8, W=9, neighborhood="rook")
G.build_edges_social_ba(m=2, rng_seed=42)
G.initial_reps(K=K)
G.fill_opinions_hbo_graph(alpha=2.0, beta=2.0, influence=0.8, scale_out=7.0)
seeds = G.choose_random_district_seeds_spaced(K=K, min_manhattan=3)
dist_labels = G.greedy_fill_districts(seeds)

# Convert to FrankenData (also attaches a full HeteroData at fd.hetero)
init_data = graph_to_frankendata(G, num_districts=K, use_scaled_opinion=True, attach_hetero=False)
print("Init_Data:", init_data)        # PyG Data with extra attribute .hetero
# print("Init_Data.hetero:", init_data.hetero) # Full HeteroData for your future GNN experiments
print("Init_Data.reps:", init_data.reps)  # list of length K-number_of_districts, each is int or None
print("Init_Data.dist_label:", init_data.dist_label)   # (N,2) float32 positions
print("Init_Data.opinion:", init_data.opinion)       # (N,1) float32 opinions
print("Init_Data.x:", init_data.x)             # (N,3) float32
print("Init_Data.social_edge:", init_data.social_edge)    # (2,E) long social edges
print("Init_Data.edge_attr:", init_data.edge_attr)     # (E,) float social edge weights
print("Init_Data.geographical_edge:", init_data.geographical_edge) # (2,E) long geo edges or None
print("Init_Data.geo_edge_attr:", init_data.geo_edge_attr)  # (E,) float geo edge weights or None
print("Init_Data.num_nodes:", init_data.num_nodes)     # Number of nodes
print("Init_Data.num_edges:", init_data.num_edges)     # Number of edges
print("Init_Data.num_districts:", K)        # Number of districts
print("Init_Data.pos:", init_data.pos)         # (N,2) float32 positions

# Sanity: you now have both worlds

assert isinstance(init_data, Data)
# assert hasattr(init_data, "hetero") and isinstance(init_data.hetero, HeteroData)

# Hand to env
env = FrankenmanderingEnv(num_voters=len(G.df_nodes), num_districts=K, FrankenData=init_data)
state, _ = env.reset()
print("state:", state)           # PyG Data with extra attribute .hetero
# print("state.hetero:", state.hetero)    # Full HeteroData for your future GNN experiments
print("state.reps:", state.reps)      # list of length K-number_of_districts, each is int or None
