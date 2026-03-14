# combined/gerry_revrecom.py
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Iterator, Optional

import numpy as np
import torch

# Requires: pip install gerrychain
from gerrychain import Graph as GCGraph
from gerrychain.partition import Partition
from gerrychain.proposals import reversible_recom
from gerrychain.tree import recursive_tree_part
from gerrychain.updaters import Tally, cut_edges


@dataclass
class GerryRevReComConfig:
    pop_col: str = "population"
    epsilon: float = 0.03          # population tolerance for reversible_recom
    M: int = 30                    # balance-edge upper bound (same meaning as your M)
    repeat_until_valid: bool = False
    seed: Optional[int] = None     # if set, seeds Python's RNG via np.random + random
    use_recursive_seed: bool = True  # use GerryChain recursive tree seeding for the initial plan


def _labels_from_fd(fd) -> np.ndarray:
    # fd.dist_label is a torch tensor of shape (N,)
    return fd.dist_label.detach().cpu().numpy().astype(np.int64, copy=True)


def _set_fd_labels(fd, labels: np.ndarray) -> None:
    fd.dist_label = torch.as_tensor(labels, dtype=torch.long)

def _dense_reindex_zero_based(labels: np.ndarray) -> tuple[np.ndarray, int]:
    uniq = np.unique(labels)
    if uniq.min() < 0:
        raise ValueError(f"Negative labels found (min={uniq.min()}).")
    remap = {int(a): i for i, a in enumerate(uniq.tolist())}
    out = np.fromiter((remap[int(a)] for a in labels), count=labels.size, dtype=np.int64)
    return out, len(uniq)

def _build_gc_graph_from_G(G, pop_col: str, pops: Optional[np.ndarray] = None) -> GCGraph:
    """
    Build a GerryChain Graph using GEO adjacency.
    Assumptions (true for your grid setup):
      - node ids are 0..N-1
      - G.df_edges_geo has columns u,v (0-based)
    """
    N = len(G.df_nodes)

    # Default population: 1 per node (matches your current Rust wrapper usage)
    if pops is None:
        pops = np.ones(N, dtype=np.int64)
    else:
        pops = np.asarray(pops, dtype=np.int64)
        if pops.shape != (N,):
            raise ValueError(f"pops must have shape (N,), got {pops.shape} with N={N}")

    # If you already have a networkx GEO graph (you do: G.G_geo), reuse it to preserve attributes.
    # GerryChain can wrap a networkx graph.
    nxg = getattr(G, "G_geo", None)
    if nxg is None:
        raise ValueError("G.G_geo not found; expected a networkx GEO graph on G.")

    # Ensure node set is exactly 0..N-1
    # (If not, you need a relabel step here.)
    missing = sorted(set(range(N)) - set(nxg.nodes))
    if missing:
        raise ValueError(f"G.G_geo is missing node ids (example: {missing[:10]})")

    # Attach population attribute
    for i in range(N):
        nxg.nodes[i][pop_col] = int(pops[i])

    return GCGraph.from_networkx(nxg)

def _build_partition(gc_graph: GCGraph, labels0: np.ndarray, pop_col: str) -> Partition:
    labels0 = np.asarray(labels0, dtype=np.int64)
    labels, K = _dense_reindex_zero_based(labels0)

    # Partition expects an assignment mapping node -> district label
    assignment = {i: int(labels[i]) for i in range(labels.size)}

    updaters = {
        "population": Tally(pop_col, alias="population"),
        "cut_edges": cut_edges,
    }
    part = Partition(gc_graph, assignment=assignment, updaters=updaters)

    # Basic sanity (no empty districts)
    cnt = np.bincount(labels, minlength=K)
    if (cnt == 0).any():
        empties = np.where(cnt == 0)[0].tolist()
        raise ValueError(f"Empty districts in seed assignment: {empties}")

    return part


def _recursive_tree_assignment(
    gc_graph: GCGraph,
    K: int,
    ideal_population: float,
    epsilon: float,
    pop_col: str,
) -> np.ndarray:
    """Create a contiguous assignment via recursive tree partitioning."""
    params = {
        "parts": list(range(K)),
        "pop_target": ideal_population,
        "pop_col": pop_col,
        "epsilon": epsilon,
    }
    assignment = recursive_tree_part(gc_graph, **params)
    return np.asarray([assignment[i] for i in range(len(assignment))], dtype=np.int64)

def _assignment_vector(partition: Partition, N: int) -> np.ndarray:
    # Guaranteed order 0..N-1
    return np.asarray([partition.assignment.mapping[i] for i in range(N)], dtype=np.int64)

def gerry_revrecom_generator(
    G,
    fd,
    *,
    total_steps: int,
    cfg: GerryRevReComConfig = GerryRevReComConfig(),
    pops: Optional[np.ndarray] = None,
    include_start: bool = False,
) -> Iterator[np.ndarray]:
    """
    Yield *accepted* reversible-ReCom plans as 0-based labels (np.int64, shape (N,)).
    Self-loops (proposal returns same Partition) are skipped.

    Note: reversible_recom uses Python's random module internally. If you care about
    reproducibility, seed both random and numpy before running.
    """
    if cfg.seed is not None:
        import random
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)

    N = len(G.df_nodes)

    labels0 = _labels_from_fd(fd)
    K = int(labels0.max() + 1)
    gc_graph = _build_gc_graph_from_G(G, cfg.pop_col, pops=pops)
    total_pop = sum(gc_graph.nodes[i][cfg.pop_col] for i in range(N))
    pop_target = total_pop / K

    if cfg.use_recursive_seed:
        labels0 = _recursive_tree_assignment(
            gc_graph,
            K,
            pop_target,
            cfg.epsilon,
            cfg.pop_col,
        )
        _set_fd_labels(fd, labels0)

    partition = _build_partition(gc_graph, labels0, cfg.pop_col)

    proposal = partial(
        reversible_recom,
        pop_col=cfg.pop_col,
        pop_target=pop_target,
        epsilon=float(cfg.epsilon),
        M=int(cfg.M),
        repeat_until_valid=bool(cfg.repeat_until_valid),
    )

    if include_start:
        start = _assignment_vector(partition, N)
        _set_fd_labels(fd, start)
        yield start

    accepted = 0
    while accepted < int(total_steps):
        new_part = proposal(partition)

        # In GerryChain's reversible_recom, self-loop returns the same object
        if new_part is partition:
            continue

        partition = new_part
        labels = _assignment_vector(partition, N)

        _set_fd_labels(fd, labels)
        yield labels
        accepted += 1
