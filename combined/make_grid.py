import pandas as pd
import numpy as np
import networkx as nx
import torch
from torch_geometric.data import HeteroData
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (ensures 3D projection)
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.tri as mtri
from matplotlib.colors import TwoSlopeNorm
from torch_geometric.utils import is_undirected, to_undirected

class Graph:
    def __init__(self, df_nodes: pd.DataFrame, G: nx.Graph):
        self.df_nodes = df_nodes
        self.G = G # keep as optional "union" graph for convenience/visualization
        self.df_edges: pd.DataFrame | None = None
        
        # NEW: separate layers
        self.G_geo = nx.Graph()
        self.G_social = nx.Graph()

        self.df_edges_geo: pd.DataFrame | None = None
        self.df_edges_social: pd.DataFrame | None = None
        
    @staticmethod
    def make_node_ids(N: int):
        """Make node ids 0..N-1"""
        ids = np.arange(N, dtype=np.int32)
        df = pd.DataFrame({"id": ids})
        G = nx.empty_graph(N)  # Create an empty graph with N nodes
        return df, G
    
    def generate_positions(self, *, 
        mode: str = "grid",# "grid" | "masked_lattice" | "uniform_box" | "uniform_polygon" | "spring_layout"
        N: int | None = None, H: int | None = None, W: int | None = None,
        bounds: tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0),
        mask_fn=None, polygon=None, seed: int = 0):
        rng = np.random.default_rng(seed)

        def _ensure_graph_size(m: int):
            # keep ids contiguous 0..m-1 in both df and G
            self.df_nodes = pd.DataFrame({"id": np.arange(m, dtype=np.int32)})
            assert (self.df_nodes.id.values == np.arange(len(self.df_nodes))).all()
            # keep separate graphs + optional union
            self.G_geo = nx.empty_graph(m)
            self.G_social = nx.empty_graph(m)
            self.G = nx.empty_graph(m)
            # reset old single-DF if present
            self.df_edges = None
            self.df_edges_geo = None
            self.df_edges_social = None

        # --- Generate (x, y) according to mode ---
        if mode == "grid":
            assert H is not None and W is not None, "grid requires H and W"
            M = H * W
            if len(self.df_nodes) != M or self.G.number_of_nodes() != M:
                _ensure_graph_size(M)
            xs, ys = np.meshgrid(np.arange(W), np.arange(H))
            x = xs.ravel().astype(np.float32)
            y = ys.ravel().astype(np.float32)

        elif mode == "masked_lattice":
            assert H is not None and W is not None and callable(mask_fn), "masked_lattice needs H,W and mask_fn"
            xs, ys = np.meshgrid(np.arange(W), np.arange(H))
            x_all = xs.ravel().astype(np.float32)
            y_all = ys.ravel().astype(np.float32)
            keep = np.fromiter((mask_fn(float(xx), float(yy)) for xx, yy in zip(x_all, y_all)), dtype=bool)
            x = x_all[keep]; y = y_all[keep]
            M = x.size
            if len(self.df_nodes) != M or self.G.number_of_nodes() != M:
                _ensure_graph_size(M)

        elif mode == "uniform_box":
            N = int(N or len(self.df_nodes) or self.G.number_of_nodes() or 0)
            assert N > 0, "uniform_box needs N"
            if len(self.df_nodes) != N or self.G.number_of_nodes() != N:
                _ensure_graph_size(N)
            xmin, xmax, ymin, ymax = bounds
            x = rng.uniform(xmin, xmax, size=N).astype(np.float32)
            y = rng.uniform(ymin, ymax, size=N).astype(np.float32)

        elif mode == "uniform_polygon":
            from shapely.geometry import Point
            assert polygon is not None, "uniform_polygon needs a shapely Polygon"
            N = int(N or len(self.df_nodes) or self.G.number_of_nodes() or 0)
            assert N > 0, "uniform_polygon needs N"
            if len(self.df_nodes) != N or self.G.number_of_nodes() != N:
                _ensure_graph_size(N)
            xmin, ymin, xmax, ymax = polygon.bounds
            xs_, ys_ = [], []
            while len(xs_) < N:
                xx = rng.uniform(xmin, xmax); yy = rng.uniform(ymin, ymax)
                if polygon.contains(Point(xx, yy)):
                    xs_.append(xx); ys_.append(yy)
            x = np.array(xs_, dtype=np.float32); y = np.array(ys_, dtype=np.float32)

        elif mode == "spring_layout":
            # Works even for empty_graph(N); ensure node set matches df
            if self.G.number_of_nodes() != len(self.df_nodes):
                _ensure_graph_size(len(self.df_nodes))
            pos = nx.spring_layout(self.G, seed=seed, dim=2)
            x = np.array([pos[i][0] for i in self.df_nodes.id], dtype=np.float32)
            y = np.array([pos[i][1] for i in self.df_nodes.id], dtype=np.float32)
            
        elif mode == "custom_layout":
            # Implement your custom layout logic here until then raise error
            raise NotImplementedError("custom_layout mode not implemented yet")
        

        else:
            raise ValueError(f"Unknown mode={mode!r}")

        # --- Validate then attach ---
        assert (self.df_nodes.id.values == np.arange(len(self.df_nodes))).all()
        assert len(x) == len(self.df_nodes) == self.G.number_of_nodes(), "length mismatch (x/y vs node set)"
        assert np.isfinite(x).all() and np.isfinite(y).all(), "non-finite coordinates"
        return self.attach_positions(x, y)

    def _push_node_attrs(self, *, to_geo=True, to_social=True, to_union=True):
        """Copy x,y (+ district/opinion if present) from df_nodes onto G, G_geo, G_social."""
        import networkx as nx
        cols = self.df_nodes.columns
        have_x = 'x' in cols; have_y = 'y' in cols
        have_d = 'district' in cols; have_o = 'opinion' in cols

        attrs = {}
        for i, row in self.df_nodes.iterrows():
            a = {}
            if have_x: a['x'] = float(row['x'])
            if have_y: a['y'] = float(row['y'])
            if have_d: a['district'] = int(row['district'])
            if have_o: a['opinion']  = float(row['opinion'])
            if a: attrs[int(row['id'])] = a

        if not attrs:
            return

        if to_union and hasattr(self, "G") and self.G is not None:
            nx.set_node_attributes(self.G, attrs)
        if to_geo and hasattr(self, "G_geo") and self.G_geo is not None:
            nx.set_node_attributes(self.G_geo, attrs)
        if to_social and hasattr(self, "G_social") and self.G_social is not None:
            nx.set_node_attributes(self.G_social, attrs)
    
    def attach_positions(self, x, y):
        # Coerce to arrays and validate length
        x_arr = np.asarray(x)
        y_arr = np.asarray(y)
        N = len(self.df_nodes)
        if x_arr.shape[0] != N or y_arr.shape[0] != N:
            raise ValueError(f"Length of x/y ({x_arr.shape[0]}/{y_arr.shape[0]}) must match number of nodes ({N})")

        self.df_nodes = self.df_nodes.assign(x=x_arr, y=y_arr)

        # Ensure the graph has the nodes present in df.id. If not, add them.
        ids = [int(i) for i in self.df_nodes.id]
        for G_ in (self.G, self.G_geo, self.G_social):
            missing = set(ids) - set(G_.nodes)
            if missing:
                G_.add_nodes_from(missing)

        # Attach attributes (ensure numeric types)
        # positions = {int(i): {"x": float(xx), "y": float(yy)} for i, xx, yy in zip(self.df_nodes.id, self.df_nodes.x, self.df_nodes.y)}
        # The above line becomes redundant with the loop below, but kept for clarity. 
        self._push_node_attrs(to_geo=True, to_social=True, to_union=True)
        return self
    
    def build_edges_grid(self, H: int | None = None, W: int | None = None, *,
        neighborhood: str = "rook",   # "rook" (4-neigh) or "queen" (8-neigh)
        weight_grid: float = 1.0, barrier_flag: int = 0):
        """
        This edge builder can handle both full grids and masked lattices. But it is only for 
        grids/lattices where nodes have integer (x,y) coordinates. When the graph is not a grid
        or masked lattice, use a different edge builder. You can have position independent edges but for 
        grids/lattices, edges depend on (x,y) coordinates.
        If H and W are provided → build a full HxW lattice.
        Else infer edges from the integer (x,y) pairs in self.df_nodes (works for masked lattices).
        Edges are undirected and de-duplicated; stored as (u,v) by node id.
        """
        assert neighborhood in ("rook", "queen")
        rows = []

        if H is not None and W is not None:
            # Full rectangular lattice
            def id_from_xy(x, y): return y * W + x

            # right edges
            for y in range(H):
                for x in range(W - 1):
                    u = id_from_xy(x,   y); v = id_from_xy(x+1, y)
                    rows.append((min(u, v), max(u, v)))

            # up edges
            for y in range(H - 1):
                for x in range(W):
                    u = id_from_xy(x, y); v = id_from_xy(x, y+1)
                    rows.append((min(u, v), max(u, v)))

            if neighborhood == "queen":
                # diagonals
                for y in range(H - 1):
                    for x in range(W - 1):
                        u = id_from_xy(x, y)
                        rows.append((min(u, id_from_xy(x+1, y+1)), max(u, id_from_xy(x+1, y+1))))
                        rows.append((min(id_from_xy(x+1, y), id_from_xy(x, y+1)),
                                    max(id_from_xy(x+1, y), id_from_xy(x, y+1))))
            edges = pd.DataFrame(rows, columns=["u", "v"]).drop_duplicates()

            # Integrity (only for full grids where E is known)
            if neighborhood == "rook":
                E_expected = H * (W - 1) + W * (H - 1)
            else:  # queen adds 2 diagonals per interior cell
                E_expected = H * (W - 1) + W * (H - 1) + 2 * (H - 1) * (W - 1)
            assert len(edges) == E_expected, "edge count mismatch for full grid"

        else:
            # Masked lattice (only the nodes present in df_nodes). Requires integer coords.
            df = self.df_nodes
            if not {"x", "y"}.issubset(df.columns):
                raise ValueError("df_nodes must have x,y to build masked grid edges")
            xy = df[["x","y"]].to_numpy()
            if not np.issubdtype(xy.dtype, np.integer):
                if np.allclose(xy, np.rint(xy)):
                    xy = np.rint(xy).astype(int)         # coerce integer-like floats
                else:
                    raise ValueError("masked lattice expects integer-like x,y (e.g., 0,1,2)")

            # map (x,y) → id for O(1) neighbor checks
            lookup = {(int(x), int(y)): int(i) for i, x, y in zip(df["id"], df["x"], df["y"])}

            # neighbor deltas
            deltas = [(1,0), (0,1), (-1,0), (0,-1)]  # rook
            if neighborhood == "queen":
                deltas += [(1,1), (1,-1), (-1,1), (-1,-1)]

            for (x, y), u in lookup.items():
                for dx, dy in deltas:
                    v = lookup.get((x+dx, y+dy))
                    if v is not None and u < v:   # undirected, dedupe
                        rows.append((u, v))

            edges = pd.DataFrame(rows, columns=["u", "v"]).drop_duplicates()
            # No fixed E_expected here—masking determines the count.

        # annotate and update the NX graph
        edges["weight_grid"]  = weight_grid
        edges["barrier_flag"] = barrier_flag
        edges["edge_type_geo"] = 1
        # --- NX graph update
        # Undirected nx.Graph only needs one direction; adding the canonical (u < v) set is enough.
        # self.G.clear_edges()  # if you want to preserve social edges in G, remove this clear
        self.G_geo.add_edges_from(edges[["u", "v"]].itertuples(index=False, name=None))

        # persist to instance (GEO table)
        self.df_edges_geo = edges.copy()
        self._push_node_attrs(to_geo=True, to_social=True, to_union=True)
        return self                                                       # ← enable chaining
    
    def fill_opinions_hbo_graph(self, *, 
        alpha: float = 2.0,         # prior Beta(α, β)
        beta: float = 2.0,
        influence: float = 0.8,     # ρ in your description (0..1)
        rng_seed: int = 42,
        scale_out: float | None = None,   # e.g. 7.0 to also write a scaled column
        scaled_colname: str = "opinion_scaled"):
        """
        Layout-agnostic HBO:
        - Uniformly sample from ALL unfilled nodes.
        - If any neighbors are already filled, compute v̄ (mean of their opinions),
        tilt a Beta toward v̄ (same a/b logic you used), draw X, then set v = (1-ρ)X + ρ v̄.
        - If no filled neighbors, draw from prior Beta(α, β).
        - Repeat until every node is assigned.

        Requires that self.G already has edges (G.neighbors drives the 'rook' notion
        when you built a rook grid; works for masked grids and non-grids too).
        """
        rng = np.random.default_rng(rng_seed)
        N = len(self.df_nodes)
        assert self.G_geo.number_of_nodes() == N, "Graph node count must match df_nodes"

        vals = np.full(N, np.nan, dtype=np.float32)
        assigned = np.zeros(N, dtype=bool)

        # Keep an array of all unfilled ids and shrink it in O(1) each draw
        unfilled = np.arange(N, dtype=np.int32)
        front = N  # active prefix length

        while front > 0:
            # Choose an index r uniformly from [0, front)
            r = int(rng.integers(0, front))
            i = int(unfilled[r])
            # Remove i from U by swapping with last active and shrinking 'front'
            unfilled[r], unfilled[front-1] = unfilled[front-1], unfilled[r]
            front -= 1

            if assigned[i]:
                continue

            # Collect already-FILLED graph neighbors' opinions
            neigh_vals = [vals[v] for v in self.G_geo.neighbors(i) if assigned[v]]

            if not neigh_vals:
                v = rng.beta(alpha, beta)
            else:
                vbar = float(np.mean(neigh_vals))
                # same "tilt" logic as your grid HBO
                if vbar > 0.5:
                    a = 2 * (1.0 + influence)
                    b = 2 * (1.0 - influence)
                else:
                    a = 2 * (1.0 - influence)
                    b = 2 * (1.0 + influence)
                draw = rng.beta(a, b)
                v = (1.0 - influence) * draw + influence * vbar
                #print("v_before_scale:", v)
                
            vals[i] = np.float32(np.clip(v, 0.0, 7.0))
            assigned[i] = True

        # write back to df_nodes and graph
        self.df_nodes = self.df_nodes.assign(opinion=vals)
        if scale_out is not None:
            self.df_nodes[scaled_colname] = np.float32(np.clip(vals * float(scale_out), 0.0, float(scale_out)))
        self._push_node_attrs(to_geo=True, to_social=True, to_union=True)
        return self

    # --------------------------- District Maps -------------------------
    def _manhattan_xy(self, i: int, j: int) -> int:
        """Manhattan distance using integer-like (x,y) on df_nodes."""
        xi, yi = float(self.df_nodes.loc[i, "x"]), float(self.df_nodes.loc[i, "y"])
        xj, yj = float(self.df_nodes.loc[j, "x"]), float(self.df_nodes.loc[j, "y"])
        return int(abs(round(xi) - round(xj)) + abs(round(yi) - round(yj)))

    def choose_random_district_seeds_spaced(self, K: int, *, min_manhattan: int = 3,
                                            rng_seed: int | None = None, max_tries: int = 10000) -> list[int]:
        """
        Randomly choose K node ids with pairwise Manhattan distance >= min_manhattan.
        On your grid, '≥2 apart' => set min_manhattan=3 to forbid distances 0,1,2.
        """
        rng = np.random.default_rng(rng_seed)
        ids = self.df_nodes["id"].to_numpy()
        seeds: list[int] = []
        tries = 0
        while len(seeds) < K and tries < max_tries:
            cand = int(rng.choice(ids))
            if all(self._manhattan_xy(cand, s) >= min_manhattan for s in seeds):
                seeds.append(cand)
            tries += 1
        if len(seeds) < K:
            raise RuntimeError(f"Could not place {K} seeds with Manhattan >= {min_manhattan}. "
                            f"Placed {len(seeds)}. Try smaller K or smaller constraint.")
        return seeds

    def greedy_fill_districts(self, seed_ids: list[int], *, rng_seed: int | None = None) -> np.ndarray:
        """
        Greedy districting (§2.2, eqs. (6)-(10)) over the GEO graph:
        - frontier N_j(t): unassigned geo-neighbors of district j
        - choose district with prob ∝ |N_j(t)|
        - claim a uniform random cell from that frontier
        Returns: labels array L (shape N,) with values in {1..K}; also writes df_nodes['district'].
        """
        assert self.df_edges_geo is not None, "Build geo edges first"
        # Strictly enforce geo-only neighbors in case social edges are overlaid later
        # (edges built by build_edges_grid are tagged edge_type='geo')
        # If your G currently has only geo edges, this is already satisfied.
        K = len(seed_ids)
        N = len(self.df_nodes)
        rng = np.random.default_rng(rng_seed)

        # State (use -1 only internally while constructing; final is 0..K-1)
        labels = np.full(N, -1, dtype=np.int32) 
        assigned = np.zeros(N, dtype=bool)

        # Initialize:  place seeds with labels 0..K-1
        for j, nid in enumerate(seed_ids):
            labels[nid] = j
            assigned[nid] = True

        # Build initial frontiers N_j(t) from G_geo.neighbors (geo graph)
        frontiers = [set() for _ in range(K)]  # indices 0..K-1
        for j, nid in enumerate(seed_ids):
            for u in self.G_geo.neighbors(nid):
                if not assigned[u]:
                    frontiers[j].add(u)

        n_assigned = len(seed_ids)

        while n_assigned < N:
            sizes = np.array([len(frontiers[j]) for j in range(K)], dtype=float)
            S = sizes.sum()
            if S == 0:
                # Degenerate fallback: attach any unassigned node that touches any district
                # (rare with rook grids). Keeps contiguity.
                attached = False
                for u in range(N):
                    if assigned[u]:
                        continue
                    # If any neighbor is assigned, adopt that district
                    for v in self.G_geo.neighbors(u):
                        if assigned[v]:
                            j = labels[v]
                            labels[u] = j
                            assigned[u] = True
                            n_assigned += 1
                            # update frontiers
                            for w in self.G_geo.neighbors(u):
                                if not assigned[w]:
                                    frontiers[j].add(w)
                            for jj in range(K):
                                frontiers[jj].discard(u)
                            attached = True
                            break
                    if attached:
                        break
                if attached:
                    continue
                # If truly isolated (shouldn't happen on a connected grid), assign arbitrarily
                u = int(np.flatnonzero(~assigned)[0])
                # nearest seed’s label:
                nearest = min(range(K), key=lambda j: abs(u - seed_ids[j]))
                labels[u] = nearest
                assigned[u] = True
                n_assigned += 1
                continue

            # Choose district with p_j ∝ |N_j(t)| (eq. 8)
            p = sizes / S
            j_choice = int(rng.choice(np.arange(K), p=p))
            # Pick a cell uniformly from that frontier (eq. 9 sampling over N_j)
            u = int(rng.choice(list(frontiers[j_choice])))
            labels[u] = j_choice
            assigned[u] = True
            n_assigned += 1

            # Update frontiers:
            #  - add unassigned neighbors of u to chosen frontier
            for w in self.G_geo.neighbors(u):
                if not assigned[w]:
                    frontiers[j_choice].add(w)
            #  - remove u from all frontiers
            for jj in range(K):
                frontiers[jj].discard(u)

        # Persist
        self.df_nodes = self.df_nodes.assign(district=pd.Series(labels, index=self.df_nodes.index).astype(np.int32))
        self.district_seeds = list(map(int, seed_ids))  # keep for social overlay reuse
        self._push_node_attrs(to_geo=True, to_social=False, to_union=True)
        return labels

    def build_edges_social_ba(self, *, m: int = 2, rng_seed: int = 42,
                          weight_social: float = 1.0) -> pd.DataFrame:
        """
        Build a Barabási–Albert SOCIAL graph with NO (u,v) normalization.

        - Keeps BA's exact orientation for edges (u,v).
        - Does NOT read/modify any GEO edge data/weights.
        - Stores/returns df_edges_social with columns: u, v, edge_type='social', weight_social.
        - Drops only exact duplicate rows (same ordered pair).

        Returns
        -------
        soc_df : pd.DataFrame with the newly generated SOCIAL edges (this run).
        """
        # --- sanity
        if getattr(self, "df_nodes", None) is None:
            raise ValueError("df_nodes missing; build nodes first.")
        
        # This checks all nodes counts. However, it is not strictly necessary for BA. If you want to use less nodes
        # you can remove this check and just manually set the value of N below.
        
        N = len(self.df_nodes)
        if self.G.number_of_nodes() != N:
            raise ValueError("Graph node count must match df_nodes.")
        if N < m + 1:
            raise ValueError(f"N={N} must be at least {m+1} for BA model")

        # --- 1) BA (undirected)
        G_ba = nx.barabasi_albert_graph(N, m, seed=rng_seed)
        E_und = G_ba.number_of_edges()
        print(f"BA (undirected) edges: {E_und} (expected m*(N-m) = {m}*({N}-{m}) = {m*(N-m)})")

        # --- 2) make it directed (both directions per pair)
        G_soc_dir = G_ba.to_directed()  # to_directed() with arcs u->v and v->u for each undirected {u,v}
        arcs = list(G_soc_dir.edges())  # list of (u, v), length = 2 * E_und
        print(f"SOC directed arcs: {len(arcs)} (should be 2 * {E_und} = {2*E_und})")

        # --- 3) persist directed edges into the SOCIAL layer DataFrame
        soc_df_dir = (
            pd.DataFrame(arcs, columns=["u", "v"]).assign(edge_type_social=1, weight_social=float(weight_social))
        )

        # --- 4) keep the directed NetworkX object on the instance (fresh graph)
        self.G_social = G_soc_dir
        self.G_social.add_edges_from(arcs)

        # --- 5) store DF and return
        self.df_edges_social = soc_df_dir
        return soc_df_dir

    def initial_reps(self, K: int):
        return None
    
    def update_union_graph(self, *, carry_layer_flags: bool = True, directed: bool = False):
        self.G = nx.DiGraph() if directed else nx.Graph()
        N = len(self.df_nodes); self.G.add_nodes_from(range(N))

        # GEO
        if directed:
            self.G.add_edges_from((u, v) for u, v in self.G_geo.edges)
            self.G.add_edges_from((v, u) for u, v in self.G_geo.edges) # make GEO behave undirected
        else:
            self.G.add_edges_from(self.G_geo.edges)

        if carry_layer_flags:
            for u, v in self.G_geo.edges:
                if directed:
                    self.G[u][v]["geo"] = 1; self.G[v][u]["geo"] = 1
                else:
                    self.G[u][v]["geo"] = 1

        # SOCIAL
        if directed:
            # keep true directionality
            self.G.add_edges_from(self.G_social.edges)
            if carry_layer_flags:
                for u, v in self.G_social.edges:
                    self.G[u][v]["social"] = 1
        else:
            # presence-only union
            for u, v in self.G_social.edges:
                if not self.G.has_edge(u, v):
                    self.G.add_edge(u, v)
                if carry_layer_flags:
                    self.G[u][v]["social"] = 1

        self._push_node_attrs(to_geo=True, to_social=True, to_union=True)
        return self

    def to_pyg_hetero(self):
        data = HeteroData()
        # x := opinion (N,1)
        if 'opinion_scaled' not in self.df_nodes.columns:
            raise ValueError("df_nodes['opinion_scaled'] is missing; fill opinions before export.")
        data['node'].x = torch.tensor(self.df_nodes[['opinion_scaled']].to_numpy(), dtype=torch.float)

        # (optional) keep positions for plotting/geometry, not as features
        if {'x','y'}.issubset(self.df_nodes.columns):
            data['node'].pos = torch.tensor(self.df_nodes[['x','y']].to_numpy(), dtype=torch.float)

        # (optional) district labels as targets/aux
        if 'district' in self.df_nodes.columns:
            data['node'].district = torch.tensor(self.df_nodes['district'].to_numpy(), dtype=torch.long)
        
        data['reps'] = None  # placeholder
            
        # GEO
        if self.df_edges_geo is not None and not self.df_edges_geo.empty:
            ei_geo = torch.tensor(self.df_edges_geo[['u','v']].to_numpy().T, dtype=torch.long)
            data['node','geo','node'].edge_index = ei_geo
            if 'weight_grid' in self.df_edges_geo.columns:
                w_geo = torch.tensor(self.df_edges_geo['weight_grid'].to_numpy(), dtype=torch.float)
                data['node','geo','node'].edge_weight = w_geo

        # SOCIAL
        if self.df_edges_social is not None and not self.df_edges_social.empty:
            ei_soc = torch.tensor(self.df_edges_social[['u','v']].to_numpy().T, dtype=torch.long)
            data['node','social','node'].edge_index = ei_soc
            if 'weight_social' in self.df_edges_social.columns:
                w_soc = torch.tensor(self.df_edges_social['weight_social'].to_numpy(), dtype=torch.float)
                data['node','social','node'].edge_weight = w_soc

        # --- quick direction checks on the exported tensors
        N = len(self.df_nodes)

        # GEO: storage-level (one-per-pair → False), semantic-level (after expansion → True)
        ei = data['node','geo','node'].edge_index
        print("GEO is_undirected (edges only):", is_undirected(ei, num_nodes=N))             # expect False
        ei2 = to_undirected(ei, num_nodes=N)  # attribute-less is fine for the check
        print("GEO is_undirected (semantic via expansion):", is_undirected(ei2, num_nodes=N)) # expect True

        # SOCIAL: stored bidirectional → True; if/when weights diverge, the attr-aware check becomes False
        ei  = data['node','social','node'].edge_index
        ew  = data['node','social','node'].edge_weight
        print("SOCIAL is_undirected (edges only):", is_undirected(ei, num_nodes=N))           # expect True
        print("SOCIAL is_undirected (edges+weights):", is_undirected(ei, edge_attr=ew, num_nodes=N))
        # The last line returns True now (weights duplicated), but will turn False once (u,v) and (v,u) weights differ.
        
        return data
    
# ---------------- Example usage ----------------
# G = Graph(*Graph.make_node_ids(9))
# G.generate_positions(mode="grid", H=3, W=3)

# # GEO (rook), store both directions in the table if desired
# G.build_edges_grid(H=3, W=3, neighborhood="rook", weight_grid=1.0, barrier_flag=0)

# # SOCIAL (BA), keep or add reverse rows as you prefer
# G.build_edges_social_ba(m=2, rng_seed=7, weight_social=1.0)

# # Union only when needed (e.g., for a quick traversal)
# G.update_union_graph(carry_layer_flags=True)

# # Districting uses only GEO:
# seeds = G.choose_random_district_seeds_spaced(K=3, min_manhattan=3, rng_seed=0)
# labels = G.greedy_fill_districts(seeds, rng_seed=0)

# # Export to PyG as clean hetero relations
# data = G.to_pyg_hetero()
    
    # Example of chaining usage
    # graph = Graph(*Graph.make_node_ids(N))
    # graph.generate_positions(mode="uniform_box", N=N).build_edges_grid(H=H, W=W, neighborhood="rook")
