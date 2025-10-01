import numpy as np
import torch
from torch_geometric.data import Data, HeteroData
import gymnasium as gym
from gymnasium import spaces
from gerry_environment_chin import FrankenData, FrankenmanderingEnv
from init_graph_to_frankendata import graph_to_frankendata
from make_grid_3 import Graph

# ---- Inchworm test ----
def normalize_array(data,min_val, max_val):

  normal_array = (data - min_val) / (max_val - min_val)

  return np.round(normal_array,2)

# ---- Inchworm test ----
# Convert to FrankenData (also attaches a full HeteroData at fd.hetero)
K = 3  # num districts
# Build graph
G = Graph(*Graph.make_node_ids(36))
G.generate_positions(mode="grid", H=6, W=6)
G.build_edges_grid(H=6, W=6, neighborhood="rook")
G.build_edges_social_ba(m=2, rng_seed=42)
G.fill_opinions_hbo_graph(alpha=2.0, beta=2.0, influence=0.8, scale_out=7.0)
seeds = G.choose_random_district_seeds_spaced(K=K, min_manhattan=3)
dist_labels = G.greedy_fill_districts(seeds)

# Convert to FrankenData (also attaches a full HeteroData at fd.hetero)
init_data = graph_to_frankendata(G, num_districts=K, use_scaled_opinion=True, attach_hetero=True)
print(init_data)        # PyG Data with extra attribute .hetero
print(init_data.hetero) # Full HeteroData for your future GNN experiments

# Sanity: you now have both worlds
from torch_geometric.data import Data, HeteroData
assert isinstance(init_data, Data)
assert hasattr(init_data, "hetero") and isinstance(init_data.hetero, HeteroData)

# Hand to env
env = FrankenmanderingEnv(num_voters=len(G.df_nodes), num_districts=K, FrankenData=init_data)
state, _ = env.reset()
print(state)           # PyG Data with extra attribute .hetero
print(state.hetero)    # Full HeteroData for your future GNN experiments
