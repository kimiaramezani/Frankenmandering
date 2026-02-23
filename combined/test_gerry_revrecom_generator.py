# combined/test_gerry_revrecom_generator.py
from __future__ import annotations

import numpy as np
import networkx as nx

from graph_initiator import build_init_data
from gerry_revrecom import gerry_revrecom_generator, GerryRevReComConfig


def count_per_district(labels: np.ndarray, K: int) -> np.ndarray:
    return np.bincount(labels, minlength=K).astype(int)


def tol_covering_seed(labels: np.ndarray, K: int) -> float:
    cnt = np.bincount(labels, minlength=K)
    avg = cnt.mean()
    over = cnt.max() / avg - 1.0
    under = 1.0 - cnt.min() / avg
    return float(max(over, under))


def check_contiguity(G_geo: nx.Graph, labels: np.ndarray, K: int) -> bool:
    for d in range(K):
        nodes = [i for i, a in enumerate(labels) if a == d]
        if not nodes:
            return False
        if not nx.is_connected(G_geo.subgraph(nodes)):
            return False
    return True


def seam_length(df_edges_geo, labels: np.ndarray) -> int:
    u = df_edges_geo["u"].to_numpy()
    v = df_edges_geo["v"].to_numpy()
    return int(np.sum(labels[u] != labels[v]))


def downstream_task_example(fd, G, K: int) -> dict:
    """
    Pretend this is your model/statistics code.
    It reads the plan from fd.dist_label and computes simple stats.
    """
    labels = fd.dist_label.detach().cpu().numpy().astype(np.int64)
    cnt = count_per_district(labels, K)
    return {
        "counts": cnt.tolist(),
        "seam": seam_length(G.df_edges_geo, labels),
        "contiguous": check_contiguity(G.G_geo, labels, K),
    }


def main():
    # Build your usual grid + FrankenData
    K, H, W = 6, 6, 8
    fd, G = build_init_data(K=K, H=H, W=W, attach_hetero=False)

    # Seed plan (what FrankenData currently holds)
    seed_labels = fd.dist_label.detach().cpu().numpy().astype(np.int64)

    # IMPORTANT: your seed is usually imbalanced, so strict epsilon=0.03 may never move.
    # For this smoke test, auto-widen epsilon just so we can see accepted moves.
    eps = tol_covering_seed(seed_labels, K)

    cfg = GerryRevReComConfig(
        epsilon=eps,
        M=30,
        seed=123,
    )

    print(f"[test] N={len(seed_labels)} K={K} seed_sizes={count_per_district(seed_labels, K).tolist()} eps_used={eps:.4f}")

    # Generate a few accepted plans
    T = 10
    got = 0
    for t, plan in enumerate(gerry_revrecom_generator(G, fd, total_steps=T, cfg=cfg), start=1):
        plan = np.asarray(plan, dtype=np.int64)

        # Core invariants: shape/range
        assert plan.shape == (len(seed_labels),)
        assert plan.min() >= 0 and plan.max() < K

        # Critical contract: generator updates fd.dist_label to match the yielded plan
        fd_labels = fd.dist_label.detach().cpu().numpy().astype(np.int64)
        assert np.array_equal(fd_labels, plan), "fd.dist_label is not synced with yielded plan"

        # Example "downstream task" reading from FrankenData
        stats = downstream_task_example(fd, G, K)
        print(f"[step {t:2d}] counts={stats['counts']} seam={stats['seam']} contiguous={stats['contiguous']}")

        got += 1

    assert got == T, f"Expected {T} accepted plans, got {got}"
    print("[test] PASS")


if __name__ == "__main__":
    main()
