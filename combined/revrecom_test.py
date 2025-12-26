# tests/test_revrecom_generator.py
# Quick functional test for the RevReCom streaming generator.

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import networkx as nx
import torch

# --- repo-relative imports ----------------------------------------------------
# Run this script from the repo root (Frankenmandering). Adjust paths if needed.
REPO = Path(__file__).resolve().parents[1] if (Path(__file__).parent.name == "tests") else Path(__file__).resolve().parent
sys.path.append(str(REPO))

from make_grid import Graph                                   # :contentReference[oaicite:3]{index=3}
from init_graph_to_frankendata import graph_to_frankendata    # :contentReference[oaicite:4]{index=4}
from revrecom_connector import revrecom_generator              # :contentReference[oaicite:5]{index=5}


# ------------------------- helpers / assertions ------------------------------

def count_per_district(labels: np.ndarray, K: int) -> np.ndarray:
    cnt = np.bincount(labels, minlength=K).astype(int)
    return cnt

def check_balance(cnt: np.ndarray, tol: float) -> tuple[bool, tuple[int,int,int]]:
    """Return (ok, (min_allowed, max_allowed, avg)) for easy printing."""
    N = int(cnt.sum()); K = int(len(cnt))
    avg = N / K
    min_allowed = int(np.floor((1.0 - tol) * avg))
    max_allowed = int(np.ceil((1.0 + tol) * avg))
    ok = (cnt >= min_allowed).all() and (cnt <= max_allowed).all()
    return ok, (min_allowed, max_allowed, int(round(avg)))

def check_contiguity(G_geo: nx.Graph, labels: np.ndarray, K: int) -> bool:
    """Each district's induced subgraph should be connected (non-empty)."""
    for d in range(K):
        nodes = [i for i, a in enumerate(labels) if a == d]
        if not nodes:
            return False  # empty district shouldn't happen
        sg = G_geo.subgraph(nodes)
        if not nx.is_connected(sg):
            return False
    return True

def cut_edges(G_edges_geo_df, labels: np.ndarray) -> int:
    """Number of geo edges crossing district boundaries (simple seam length proxy)."""
    # df has columns u,v; labels is 0..K-1
    u = G_edges_geo_df["u"].to_numpy()
    v = G_edges_geo_df["v"].to_numpy()
    return int(np.sum(labels[u] != labels[v]))


# ------------------------------- main test -----------------------------------

def main():
    # 1) Build a small grid GEO graph + SOCIAL overlay + opinions + initial districts
    H, W = 6, 8              # grid shape (change freely)
    K = 6                    # number of districts
    rng_seed = 7

    # Fresh node set 0..N-1 and empty graphs
    G = Graph(*Graph.make_node_ids(H * W))                           # :contentReference[oaicite:6]{index=6}
    G.generate_positions(mode="grid", H=H, W=W, seed=rng_seed)       # :contentReference[oaicite:7]{index=7}
    G.build_edges_grid(H=H, W=W, neighborhood="rook", weight_grid=1.0,
                       barrier_flag=0, use_barrier=1)                 # :contentReference[oaicite:8]{index=8}
    # SOCIAL is required by graph_to_frankendata
    G.build_edges_social_ba(m=2, rng_seed=rng_seed, weight_social=1.0)  # :contentReference[oaicite:9]{index=9}

    # Opinions in [0..7] (scaled col required by graph_to_frankendata)
    G.fill_opinions_hbo_graph(alpha=2.0, beta=2.0, influence=0.8,
                              rng_seed=rng_seed, scale_out=7.0)       # :contentReference[oaicite:10]{index=10}

    # Seed + greedy fill 0..K-1 labels on GEO graph
    seeds = G.choose_random_district_seeds_spaced(K, min_manhattan=3, rng_seed=rng_seed)  # :contentReference[oaicite:11]{index=11}
    labels0 = G.greedy_fill_districts(seeds, rng_seed=rng_seed)                           # :contentReference[oaicite:12]{index=12}

    # Optional: union graph (not strictly required here)
    G.update_union_graph(carry_layer_flags=True, directed=False)                           # :contentReference[oaicite:13]{index=13}

    # 2) Convert to FrankenData
    fd = graph_to_frankendata(G, num_districts=K, use_scaled_opinion=True, attach_hetero=False)  # :contentReference[oaicite:14]{index=14}

    # 3) Stream T accepted RevReCom plans
    T = 50
    tol = 0.03
    M = 30
    gen = revrecom_generator(G, fd, total_steps=T, tol=tol, M=M, seed=123, threads=1, batch=64, chunk=25)  # :contentReference[oaicite:15]{index=15}

    print(f"\n--- RevReCom generator test on {H}x{W} grid, K={K}, T={T}, tol=±{tol*100:.1f}% ---")
    geo_df = G.df_edges_geo

    # Print 3 snapshots and validate every step
    first3 = []
    ok_all = True
    for t, plan in enumerate(gen, start=1):
        plan = np.asarray(plan, dtype=np.int64)

        # Validations
        cnt = count_per_district(plan, K)
        ok_bal, (mn, mx, avg) = check_balance(cnt, tol)
        ok_con = check_contiguity(G.G_geo, plan, K)
        s = cut_edges(geo_df, plan)

        ok_step = ok_bal and ok_con
        ok_all = ok_all and ok_step

        if t <= 3:
            first3.append((t, cnt.tolist(), s))
        if (t % 10) == 0:
            print(f"[step {t:3d}] balance {cnt.tolist()} (allowed {mn}..{mx}, avg≈{avg}), "
                  f"contiguous={ok_con}, seam={s}")

    print("\nFirst 3 accepted plans (counts per district, seam length):")
    for t, cnt, seam in first3:
        print(f"  step {t:2d}: counts={cnt}, seam={seam}")

    print(f"\nAll steps valid: {ok_all}")
    assert ok_all, "Some step failed balance or contiguity checks"

    # Example: if you also want all plans in memory at once
    # from revrecom_connector import revrecom_collect_all
    # all_plans = revrecom_collect_all(G, fd, total_steps=T, tol=tol, M=M, seed=123)
    # print("all_plans shape:", all_plans.shape)  # (T, N)

if __name__ == "__main__":
    main()
