# revrecom_connector.py

import numpy as np
import torch
from revrecom_py import run_revrecom

def _dense_reindex_zero_based(lbl0: np.ndarray) -> tuple[np.ndarray, int]:
    uniq = np.unique(lbl0)
    if uniq.min() < 0:
        raise ValueError(f"Negative labels found (min={uniq.min()}).")
    # map old labels -> 0..K-1
    remap = {int(a): i for i, a in enumerate(uniq.tolist())}
    out = np.fromiter((remap[int(a)] for a in lbl0), count=lbl0.size, dtype=np.int64)
    K = len(uniq)
    return out, K

def _tol_covering_seed(labels_zero_based: np.ndarray, k: int) -> float:
    """
    Minimum tolerance so the current seed plan is within [min_pop, max_pop].
    This prevents the internal ReCom buffers from being undersized at step 0.
    """
    counts = np.bincount(labels_zero_based, minlength=k)
    avg = counts.mean()
    over = counts.max() / avg - 1.0     # how far the max is above average
    under = 1.0 - counts.min() / avg    # how far the min is below average
    return float(max(over, under))

def run_rev_from_frankendata(
    G,
    fd,
    tol: float = 0.03,
    M: int = 0,               # neutral by default; set >0 only if you want that cap
    steps: int = 2000,
    seed: int = 123,
    threads: int = 8,
    batch: int = 64,
    *,
    auto_widen_tol: bool = True,
    debug_print: bool = False,
):
    """
    Run a chunk of RevReCom moves and return ALL accepted plans as lists of 0-based labels.
    The first element is the starting plan; subsequent elements are accepted plans.
    """
    N = len(G.df_nodes)

    # unique undirected 0-based geo edges
    edges = []
    for u, v in G.df_edges_geo[['u', 'v']].to_numpy():
        u, v = int(u), int(v)
        if u != v:
            edges.append((min(u, v), max(u, v)))
    edges = list(set(edges))

    # every node must appear at least once
    used = set([i for e in edges for i in e])
    missing = sorted(set(range(N)) - used)
    if missing:
        raise ValueError(f"Geo edge list missing {len(missing)} nodes, e.g. {missing[:10]}")

    # guard: no endpoint >= N
    max_seen = max((max(u, v) for (u, v) in edges), default=-1)
    if max_seen >= N:
        raise ValueError(f"Edge endpoint out of range: found node id {max_seen} with N={N} "
                         f"(edges must be 0..{N-1})")

    # DENSE reindex labels -> 0..K-1, no gaps
    lbl0 = fd.dist_label.cpu().numpy().astype(np.int64, copy=True)
    lbl, K = _dense_reindex_zero_based(lbl0)

    # no empty districts
    cnt = np.bincount(lbl, minlength=K)
    if (cnt == 0).any():
        empties = np.where(cnt == 0)[0].tolist()
        raise ValueError(f"Empty districts after reindex: {empties}")

    # pops (uniform unless you have per-node pops)
    pops = [1] * N

    # ensure tol is wide enough to contain the seed (prevents internal buffer undersize)
    if auto_widen_tol:
        tol_needed = _tol_covering_seed(lbl, K)
        if tol_needed > tol:
            tol = tol_needed + 1e-6  # tiny epsilon
    if debug_print:
        print(f"[revrecom_connector] K={K} sizes={cnt.tolist()} tol_used={tol:.4f}")

    # call Rust (wrapper will 1-base labels for the crate, then collect back 0-based)
    seq = run_revrecom(
        edges, pops, lbl.astype(np.uint32).tolist(),
        k=int(K), num_steps=int(steps), tol=float(tol),
        balance_ub=int(M), rng_seed=int(seed),
        n_threads=int(threads), batch_size=int(batch),
    )
    return seq

def revrecom_generator(
    G,
    fd,
    *,
    total_steps: int,
    tol: float = 0.03,
    M: int = 0,
    seed: int = 42,
    threads: int = 8,
    batch: int = 64,
    chunk: int = 500,
    include_start: bool = False,
    auto_widen_tol: bool = True,
    debug_print: bool = False,
):
    """
    Yield *every accepted plan* (labels as a 1D numpy array, dtype=int64, values in [0..K-1]).
    Streams in CHUNKS so you don't hold all T plans in memory at once.

    Yields
    ------
    labels : np.ndarray, shape (N,), dtype=int64
        District labels 0..K-1 for each node, one array per *accepted* move
        (and optionally the starting plan if include_start=True).
    """
    # Optionally yield the current plan first
    if include_start:
        yield fd.dist_label.cpu().numpy().astype(np.int64, copy=True)

    steps_left, s = int(total_steps), int(seed)
    while steps_left > 0:
        take = min(chunk, steps_left)
        # Run a chunk of accepted moves; seq[0] is the chunk's start, seq[1:] are new plans
        seq = run_rev_from_frankendata(
            G, fd,
            tol=tol, M=M, steps=take, seed=s,
            threads=threads, batch=batch,
            auto_widen_tol=auto_widen_tol, debug_print=debug_print,
        )

        # Yield each accepted plan and keep FrankenData in sync
        for plan in seq[1:]:
            labels = np.asarray(plan, dtype=np.int64)
            fd.dist_label = torch.as_tensor(labels, dtype=torch.long)
            yield labels

        steps_left -= take
        s += take  # advance seed so next chunk differs


def revrecom_collect_all(G, fd, *, total_steps: int, **kw) -> np.ndarray:
    """
    Convenience: collect all accepted plans into an array of shape (T, N).
    (Does NOT include the starting plan; it returns exactly total_steps rows.)
    """
    return np.stack(list(revrecom_generator(G, fd, total_steps=total_steps, **kw)), axis=0)