# init_graph_to_frankendata.py
# This script initializes a graph structure and prepares it for use with Frankendata.

import numpy as np
import torch
from torch_geometric.data import Data, HeteroData
import networkx as nx

from gerry_environment import FrankenData   # make sure path/module is correct

def graph_to_frankendata(G, 
                         num_districts: int, 
                         use_scaled_opinion: bool = True,
                         attach_hetero: bool = True) -> FrankenData:
    """
    Convert your Graph (make_grid_2.Graph) into a FrankenData object the env expects.
    - num_districts: D (must match labels present in G.df_nodes['district'])
    - use_scaled_opinion: if True, send 0..7 'opinion_scaled' as features; else send 0..1 'opinion'
    """
    N = len(G.df_nodes)
    D = int(num_districts)
    if D <= 0:
        raise ValueError("num_districts must be >= 1")

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

    # --- district labels (N,) → make ZERO-BASED for env logic and ML losses
    if 'district' not in G.df_nodes.columns:
        raise ValueError("district labels missing; run greedy_fill_districts(..) first.")
    dist_label_1K = G.df_nodes['district'].to_numpy(dtype=np.int64)
    dist_label = dist_label_1K
    if dist_label.min() < 0 or dist_label.max() >= D:
        raise ValueError(f"District labels out of range after zero-base. Got unique={np.unique(dist_label)}")

    # --- geo edges: edge_index (2,E) + edge_attr (E,) (One per pair, not directed)
    if not G.df_edges_geo.empty:
        geo_edge = torch.tensor(G.df_edges_geo[['u','v']].to_numpy().T, dtype=torch.long)
        geo_attr = torch.tensor(G.df_edges_geo['weight_grid'].to_numpy(), dtype=torch.float) \
                if 'weight_grid' in G.df_edges_geo.columns else None
    else:
        geo_edge, geo_attr = None, None
    
    # --- SOCIAL edges: edge_index (2,E) + edge_attr (E,)
    if G.df_edges_social is None or G.df_edges_social.empty:
        raise ValueError("No social edges found; build_edges_social_ba(..) first.")

    soc_ei = G.df_edges_social[['u','v']].to_numpy(dtype=np.int64).T  # (2,E)
    if 'weight_social' in G.df_edges_social.columns:
        edge_attr = G.df_edges_social['weight_social'].to_numpy(dtype=np.float32)
    else:
        edge_attr = np.ones(soc_ei.shape[1], dtype=np.float32)

    orig_edge_num = soc_ei.shape[1]  # env uses this to split base vs augmented

    # --- initial assignment matrix (N, D) from labels (1..D)
    # Uncomment if you are sending some initial assignment but we don't require it now.
    # Note: the env will overwrite this after each step anyway.
    # D = int(num_districts)
    # if D <= 0:
    #     raise ValueError("num_districts must be >=1")
    # # labels are 1..K; convert to 0..K-1 for one-hot
    # zero_based = dist_label - 1
    # if zero_based.min() < 0 or zero_based.max() >= D:
    #     raise ValueError(f"District labels out of range 1..{D}. Got unique={np.unique(dist_label)}")
    # assignment = np.zeros((N, D), dtype=np.float32)
    # assignment[np.arange(N), zero_based] = 1.0   # one-hot rows

    # --- set initial representatives (D,)
    reps = [-1]*D  # -1 means "not yet elected"
    
    # --- Build FrankenData (NOTE: kw names must match class __init__)
    fd = FrankenData(
        social_edge       = soc_ei,
        geographical_edge = geo_edge,
        orig_edge_num     = orig_edge_num,
        opinion           = opinion,
        pos               = pos,
        reps              = reps,
        dist_label        = dist_label,
        edge_attr         = edge_attr,
        geo_edge_attr     = geo_attr,
    )

    # --- (optional) also keep a full heterogeneous graph inside the Data
    if attach_hetero:
        fd.hetero = G.to_pyg_hetero()  # a torch_geometric.data.HeteroData
    return fd


def inchworm_to_frankendata(G_nx, num_districts: int | None = None):
    """
    Minimal adapter: NetworkX (inchworm) -> FrankenData
    - Social edges/weights: passed as None
    - Geo edges/weights: taken from G_nx edges; weight from 'weight_grid' if present else 1.0
    - District labels: assume 1..K on nodes; convert to 0..K-1
    - Opinion: take node attr 'opinion' (default 0.0); shape (N,1), float32
    - Pos: take node attrs 'x','y' if present; else deterministic spring layout
    - orig_edge_num = 0
    """
    # ----- stable node order → ids 0..N-1
    nodes = sorted(G_nx.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    N = len(nodes)

    # ----- dist_label (zero-based)
    dist = np.asarray([G_nx.nodes[n].get("district", 1) for n in nodes], dtype=np.int64)
    K = int(dist.max()) if num_districts is None else int(num_districts)
    dist_label = (dist - 1).astype(np.int64)

    # ----- opinion (N,1) float32
    opin = np.asarray([G_nx.nodes[n].get("opinion", 0.0) for n in nodes], dtype=np.float32)
    opinion = opin[:, None]  # (N,1)

    # ----- pos (N,2) float32
    has_xy = all(("x" in G_nx.nodes[n] and "y" in G_nx.nodes[n]) for n in nodes)
    if has_xy:
        pos = np.stack([[G_nx.nodes[n]["x"], G_nx.nodes[n]["y"]] for n in nodes], axis=0).astype(np.float32)
    else:
        layout = nx.spring_layout(G_nx, seed=0, dim=2)
        pos = np.stack([layout[n] for n in nodes], axis=0).astype(np.float32)

    # ----- geographical_edge (2, E_geo) long, geo_edge_attr (E_geo,) float32
    undirected_pairs = set()
    for u, v in G_nx.edges():
        a, b = idx[u], idx[v]
        if a == b:
            continue
        if a > b:
            a, b = b, a
        undirected_pairs.add((a, b))
    pairs = sorted(undirected_pairs)

    if len(pairs):
        geographical_edge = np.asarray(pairs, dtype=np.int64).T  # (2, E_geo)
        geo_w = []
        for a, b in pairs:
            u, v = nodes[a], nodes[b]
            w = G_nx[u][v].get("weight_grid", 1.0)
            geo_w.append(float(w))
        geo_edge_attr = np.asarray(geo_w, dtype=np.float32)
    else:
        geographical_edge = np.empty((2, 0), dtype=np.int64)
        geo_edge_attr = np.empty((0,), dtype=np.float32)

    # ----- social edges/weights: None (you requested passing None through)
    social_edge = np.empty((2, 0), dtype=np.int64)
    edge_attr = np.empty((0,), dtype=np.float32)
    orig_edge_num = 0
    reps = [-1] * K

    # ----- build FrankenData (cast only where necessary)
    fd_inch = FrankenData(
        social_edge=social_edge,                   # None
        geographical_edge=geographical_edge,       # (2, E_geo)
        orig_edge_num=int(orig_edge_num),          # 0
        opinion=opinion,                           # (N,1)
        pos=pos,                                   # (N,2)
        reps=reps,                                 # list length K
        dist_label=dist_label,                     # (N,)
        edge_attr=edge_attr,                       # None
        geo_edge_attr=geo_edge_attr,               # (E_geo,)
    )
    return fd_inch
