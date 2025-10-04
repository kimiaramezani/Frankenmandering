import numpy as np
import torch
from torch_geometric.data import Data, HeteroData
import gymnasium as gym
from gymnasium import spaces
from gerry_environment import FrankenData, FrankenmanderingEnv
from init_graph_to_frankendata import graph_to_frankendata, inchworm_to_frankendata
from make_grid import Graph
from torch_geometric.data import Data, HeteroData
import networkx as nx

# ---- General graph builder and Frankendata converter ----
# Convert to FrankenData (also attaches a full HeteroData at fd.hetero)
# We will later write a generator function which yields different FrankenData objects for different districts 
# and number of voters (Nodes).

def build_init_data(
    K: int = 6,
    H: int = 8,
    W: int = 9,
    ba_m: int = 2,
    rng_seed: int = 42,
    neighborhood: str = "rook",
    hbo_alpha: float = 2.0,
    hbo_beta: float = 2.0,
    hbo_influence: float = 0.8,
    scale_out: float = 7.0,
    min_manhattan: int = 3,
    attach_hetero: bool = False,
    use_scaled_opinion: bool = True,
):
    """
    Build the exact graph used in your initiator snippet and return FrankenData.

    Steps:
      1) N = H * W nodes, grid positions
      2) GEO edges (rook/queen)
      3) SOCIAL edges via Barabási–Albert (m=ba_m)
      4) Initialize reps (placeholder per your current Graph API)
      5) Fill HBO opinions (optionally writing 'opinion_scaled' via scale_out)
      6) District seeds spaced by Manhattan distance, then greedy fill
      7) Convert to FrankenData with graph_to_frankendata

    Returns:
      init_data  : FrankenData (PyG Data subclass your env expects)
      G          : the built Graph object (for any downstream use)
    """
    N = H * W

    # 1) nodes & positions
    G = Graph(*Graph.make_node_ids(N))
    G.generate_positions(mode="grid", H=H, W=W)

    # 2) GEO edges
    G.build_edges_grid(H=H, W=W, neighborhood=neighborhood)

    # 3) SOCIAL edges
    G.build_edges_social_ba(m=ba_m, rng_seed=rng_seed)

    # 4) reps init (kept to mirror the pipeline)
    G.initial_reps(K=K)

    # 5) HBO opinions (writes 'opinion' and, with scale_out, 'opinion_scaled')
    G.fill_opinions_hbo_graph(alpha=hbo_alpha, beta=hbo_beta,
                              influence=hbo_influence, scale_out=scale_out)

    # 6) District labels
    seeds = G.choose_random_district_seeds_spaced(K=K, min_manhattan=min_manhattan)
    _ = G.greedy_fill_districts(seeds)

    # 7) Convert to FrankenData (uses 'opinion_scaled' if use_scaled_opinion=True)
    init_data = graph_to_frankendata(
        G,
        num_districts=K,
        use_scaled_opinion=use_scaled_opinion,
        attach_hetero=attach_hetero,
    )

    return init_data, G

# ---- Inchworm test graph builder and Frankendata converter ----

def build_inchworm_init_data():
    """
    Build the exact inchworm test graph and return (init_data, G_inch):
      - Nodes: 0..9
      - Geo edges: as per the test structure
      - Edge weights: 'weight_grid' = 1.0
      - Node attrs: 'opinion', 'district' (districts are None; converter leaves dist_label=None)
      - Social edges: none (converter leaves social tensors empty)
    """
    G_inch = nx.Graph()
    G_inch.add_nodes_from(range(10))

    # Edges per the inchworm test structure
    edges_to_add = [
        (0,3), (1,3), (2,3),
        (3,4), (4,5), (5,6), (6,7), (7,8), (8,9),
        (3,7), (3,8), (3,9), (4,8), (4,9),
    ]
    G_inch.add_edges_from(edges_to_add)

    # Uniform geo weights
    nx.set_edge_attributes(G_inch, {e: {"weight_grid": 1.0} for e in edges_to_add})

    # Opinions (no scaling; converter only reshapes to (N,1))
    opinions = {0:0, 1:0, 2:0, 3:1, 4:2, 5:3, 6:4, 7:5, 8:5, 9:5}
    nx.set_node_attributes(G_inch, opinions, name="opinion")
    
    # --- Stash social/dist/reps on the graph (pass-through for the converter) ---
    G_inch.graph["social_edge"] = np.empty((2, 0), dtype=np.int64)
    G_inch.graph["edge_attr"]   = np.empty((0,),   dtype=np.float32)
    G_inch.graph["orig_edge_num"] = G_inch.graph["social_edge"].shape[1]  # 0 if empty
    
    
    G_inch.graph["dist_label"]  = None
    G_inch.graph["reps"]        = None
    
    # Convert to FrankenData (social edges remain empty; geo edges populated)
    init_data = inchworm_to_frankendata(G_inch)

    return init_data, G_inch

def build_inchworm_soc_init_data():
    """
    Build a 6-node inchworm test graph with:
      - GEO (undirected) edges: 0-1-2-3-4-5, weight_grid = 1.0
      - SOCIAL (directed) edges: 0→1→2→3→4→5 and 5→4→3→2→1→0, weight = 1.0
      - Opinions on nodes (no scaling)
      - dist_label=None, reps=None
    Returns (init_data, G_inch). inchworm_to_frankendata reads everything from G_inch.
    """
    G_inch = nx.Graph()
    G_inch.add_nodes_from(range(6))  # nodes 0..5

    # ---------- GEO (undirected) ----------
    geo_edges = [(i, i+1) for i in range(5)]          # 0-1, 1-2, 2-3, 3-4, 4-5
    G_inch.add_edges_from(geo_edges)
    nx.set_edge_attributes(G_inch, {e: {"weight_grid": 1.0} for e in geo_edges})

    # ---------- Opinions ----------
    opinions = {0: 0.0, 1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0, 5: 6.0}  # (fix: added missing comma)
    nx.set_node_attributes(G_inch, opinions, name="opinion")

    # ---------- SOCIAL (directed) via to_directed ----------
    # Build an undirected "spine" for social, then make it directed
    soc_edges_undirected = [(i, i+1) for i in range(5)]
    G_soc = nx.Graph()
    G_soc.add_nodes_from(range(6))
    G_soc.add_edges_from(soc_edges_undirected)
    nx.set_edge_attributes(G_soc, {e: {"weight_soc": 1.0} for e in soc_edges_undirected})

    # Each undirected edge becomes two arcs (u,v) and (v,u)
    G_soc_dir = G_soc.to_directed()

    # Convert directed edges -> arrays expected by the converter
    soc_edges_dir = list(G_soc_dir.edges())  # list[(u,v)]
    if soc_edges_dir:
        social_edge = np.asarray(soc_edges_dir, dtype=np.int64).T            # shape (2, E_soc)
        edge_attr   = np.asarray(
            [float(G_soc_dir[u][v].get("weight_soc", 1.0)) for (u, v) in soc_edges_dir],
            dtype=np.float32
        )                                                                    # shape (E_soc,)
    else:
        social_edge = np.empty((2, 0), dtype=np.int64)
        edge_attr   = np.empty((0,),   dtype=np.float32)

    # ---------- Stash social + metadata on the *same* graph ----------
    G_inch.graph["social_edge"]   = social_edge
    G_inch.graph["edge_attr"]     = edge_attr
    G_inch.graph["orig_edge_num"] = int(social_edge.shape[1])  # number of directed arcs
    G_inch.graph["dist_label"]    = None
    G_inch.graph["reps"]          = None

    # ---------- Convert (pass-through) ----------
    init_inch_soc_data = inchworm_to_frankendata(G_inch)
    return init_inch_soc_data, G_inch



