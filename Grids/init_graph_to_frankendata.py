# init_graph_to_frankendata.py
# This script initializes a graph structure and prepares it for use with Frankendata.

import numpy as np
import torch

from Environment.gerry_environment_12 import FrankenData, elect_representatives   # make sure path/module is correct

def graph_to_frankendata(G, num_districts: int, use_scaled_opinion: bool = True) -> FrankenData:
    """
    Convert your Graph (make_grid_2.Graph) into a FrankenData object the env expects.
    - num_districts: D (must match labels present in G.df_nodes['district'])
    - use_scaled_opinion: if True, send 0..7 'opinion_scaled' as features; else send 0..1 'opinion'
    """
    N = len(G.df_nodes)

    # --- positions (N,2)
    if not {'x','y'}.issubset(G.df_nodes.columns):
        raise ValueError("Graph is missing x,y positions.")
    pos = G.df_nodes[['x','y']].to_numpy(dtype=np.float32)

    # --- opinions (N, 1)
    if use_scaled_opinion:
        if 'opinion_scaled' not in G.df_nodes.columns:
            raise ValueError("opinion_scaled missing; rerun fill_opinions_hbo_graph(scale_out=7.0, ...)")
        opinion = G.df_nodes[['opinion_scaled']].to_numpy(dtype=np.float32)
    else:
        if 'opinion' not in G.df_nodes.columns:
            raise ValueError("opinion missing; run fill_opinions_hbo_graph first")
        opinion = G.df_nodes[['opinion']].to_numpy(dtype=np.float32)

    # --- district labels (N,)
    if 'district' not in G.df_nodes.columns:
        raise ValueError("district labels missing; run greedy_fill_districts(..) first.")
    dist_label = G.df_nodes['district'].to_numpy(dtype=np.int64)  # labels in {1..K}; will cast to torch.long in FrankenData

    # --- social edges: edge_index (2,E) + edge_attr (E,)
    if G.df_edges_social is None or G.df_edges_social.empty:
        raise ValueError("No social edges found; build_edges_social_ba(..) first.")
    soc = G.df_edges_social[['u','v']].to_numpy(dtype=np.int64)            # (E,2)
    so_edge = soc.T                                                        # (2,E)
    if 'weight_social' in G.df_edges_social.columns:
        edge_attr = G.df_edges_social['weight_social'].to_numpy(dtype=np.float32)  # (E,)
    else:
        edge_attr = np.ones(so_edge.shape[1], dtype=np.float32)

    orig_edge_num = so_edge.shape[1]  # base social edge count (before any augmentation in the env)

    # --- initial assignment matrix (N, D) from labels (1..D)
    D = int(num_districts)
    if D <= 0:
        raise ValueError("num_districts must be >=1")
    # labels are 1..K; convert to 0..K-1 for one-hot
    zero_based = dist_label - 1
    if zero_based.min() < 0 or zero_based.max() >= D:
        raise ValueError(f"District labels out of range 1..{D}. Got unique={np.unique(dist_label)}")
    assignment = np.zeros((N, D), dtype=np.float32)
    assignment[np.arange(N), zero_based] = 1.0   # one-hot rows

    # --- representatives (optional but useful to initialize env state)
    # Franken env has helper; we can compute now so env starts with consistent reps.
    reps = elect_representatives(zero_based, opinion, D)  # returns list of ints/None

    # Convert None to -1 (env does that later anyway)
    reps = [(-1 if r is None else int(r)) for r in reps]

    # --- Build FrankenData
    FD = FrankenData(
        so_edge    = so_edge,          # (2,E) long
        assignment = assignment,       # (N,D) float; env will replace this after each step
        orig_edge_num = orig_edge_num, # scalar
        opinion   = opinion,           # (N,1) float32
        pos       = pos,               # (N,2) float32
        reps      = reps,              # (D,) int
        dist_label= dist_label,        # (N,)  will be cast inside FrankenData
        edge_attr = edge_attr          # (E,) float32
    )
    return FD
