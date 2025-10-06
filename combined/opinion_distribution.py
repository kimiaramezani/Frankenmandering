# experiment_static_franken.py

import numpy as np
import torch
from typing import List, Tuple
from gerry_environment import FrankenmanderingEnv
from graph_initiator import build_init_data
from init_graph_to_frankendata import graph_to_frankendata
# from mcmc_baseline import labels_to_action, proposal_flip,

# -------------------- knobs --------------------
K = 6                    # districts per world f
H, W = 8, 9              # grid size for geography (N = H*W)
F = 5                   # number of worlds to sample
M_per_f = 3             # number of random districtings per world
STEPS = 100              # env steps per (f, m)

# GEO & SOCIAL configs
NEIGHBORHOOD = "rook"    # or "queen"
BA_m = 2                 # Barabási–Albert parameter
HBO_ALPHA, HBO_BETA = 2.0, 2.0
HBO_INFL = 0.8
SCALE_OUT = 7.0          # opinions in 0..7 (matches your converter default)
MIN_MANHATTAN = 3        # spacing constraint for seed selection

# DRF hyperparameters (example; tune as needed)
eps_indiff, eps_assim, eps_backfire, eps_irrel, eps_amb = 0, 3.0, 3.0, 200.0, 0.0
assim_shift, back_shift, indiff_shift, amb_shift, irr_shift = 1, -1, 0.0, 0.0, 0.0

RNG = np.random.default_rng(1234)

def labels_to_action(labels, num_districts, dtype=np.float32):
    """
    Convert an integer label vector (shape [N]) to an action matrix
    expected by env.step: shape (N, num_districts), each row is
    a 1-hot encoding of the desired district for that voter.
    """
    N = len(labels)
    A = np.zeros((N, num_districts), dtype=dtype)
    for i, lab in enumerate(labels):
        if lab >= 0 and lab < num_districts:
            A[i, int(lab)] = 1.0
        else:
            # keep row zeros -> will become -1 label in env.step (avoid if possible)
            pass
    return A

def sample_world(K: int, H: int, W: int, seed: int):
    """
    Build one world f = (geography + opinions + social).
    Returns (FrankenData, Graph) so we can regenerate many district maps on the same f.
    """
    fd, G = build_init_data(
        K=K, H=H, W=W, ba_m=BA_m, rng_seed=seed,
        neighborhood=NEIGHBORHOOD,
        hbo_alpha=HBO_ALPHA, hbo_beta=HBO_BETA, hbo_influence=HBO_INFL,
        scale_out=SCALE_OUT, min_manhattan=MIN_MANHATTAN,
        attach_hetero=False, use_scaled_opinion=True,
    )
    return fd, G

def random_maps_for_world(G, K: int, m_count: int, base_seed: int) -> List[np.ndarray]:
    """
    Generate m_count random district labelings for the same world f
    by reseeding the spaced-seed picker + greedy fill each time.
    """
    labels_list = []
    for j in range(m_count):
        seeds = G.choose_random_district_seeds_spaced(K, min_manhattan=MIN_MANHATTAN, rng_seed=base_seed + j)
        labels = G.greedy_fill_districts(seeds, rng_seed=base_seed + j)
        labels_list.append(labels.copy())
    return labels_list

def fd_with_labels(G, K: int, labels: np.ndarray):
    """
    Write labels into G.df_nodes['district'], then convert to FrankenData.
    (This fixes the initial district inside the env state.)
    """
    G.df_nodes["district"] = labels.astype(np.int32)
    return graph_to_frankendata(
        G, num_districts=K,
        use_scaled_opinion=True, attach_hetero=False
    )

def op_diff(fd, K: int, steps: int, drf_params):
    """
    Run `steps` with a static (hard) assignment equal to fd.dist_label.
    Collect final distance-to-ideal (sum of L2 norms to c*).
    Also prints a few per-step magnitudes for sanity.
    """
    
    # final distance to c* (env stores c*; opinions are in obs.opinion)
    N, m = fd.opinion.shape
    c_star = np.full((N, m), 0.0, dtype=np.float32)   # your custom ideal

    env = FrankenmanderingEnv(
        num_voters=fd.opinion.shape[0],
        num_districts=K,
        opinion_dim=fd.opinion.shape[1],
        horizon=steps,
        seed=0,
        FrankenData=fd,
        target_opinion=c_star,  # defaults to zeros of shape (N, m)
    )

    # static one-hot action from initial labels
    labels = np.asarray(fd.dist_label, dtype=np.int32)
    action = labels_to_action(labels, K)

    obs, info = env.reset()

    # ---- DEBUG PEEK: t=0 (before any update) ----
    x_t = np.asarray(obs.opinion)
    step_sum = np.linalg.norm(x_t - c_star, axis=1).sum() / (env.num_voters)
    print(f"t= 0 (pre-step)  sum|x|={step_sum:.3f}")

    for t in range(steps):
        obs, reward, terminated, truncated, info = env.step(action, *drf_params)

        # ---- DEBUG PEEK: after this step ----
        # choose any checkpoints you want; these hit early/mid/last
        if t in (0, steps//2 - 1, steps - 1):
            x_t = np.asarray(obs.opinion)
            step_sum = np.linalg.norm(x_t - c_star, axis=1).sum() / (env.num_voters)
            print(f"t={t+1:3d} (post-step) sum|x|={step_sum:.3f}")

        if terminated or truncated:
            break

    x_final = np.asarray(obs.opinion)
    print(f"x_final sample: {x_final.shape} ...")
    # final distance to c*
    final_dist = (np.linalg.norm(x_final - c_star, axis=1).sum())/(env.num_voters)

    # Histogram of the averages across runs
    return float(final_dist)

def main():
    drf_params = (
        eps_indiff, eps_assim, eps_backfire, eps_irrel, eps_amb,
        assim_shift, back_shift, indiff_shift, amb_shift, irr_shift
    )
    results = []   # list of final distances across all (f, m)
    meta = []      # optional: (f_id, m_id)

    for f_id in range(F):
        seed = int(RNG.integers(0, 10_000_000))
        fd_world, G = sample_world(K, H, W, seed)  # f
        # generate M maps for this world
        labels_list = random_maps_for_world(G, K, M_per_f, base_seed=seed + 1000)

        for m_id, labels in enumerate(labels_list):
            fd = fd_with_labels(G, K, labels)      # (f, m) as FrankenData
            d = op_diff(fd, K, STEPS, drf_params)
            results.append(d)
            meta.append((f_id, m_id))

    results = np.asarray(results, dtype=np.float64)
    print(f"Samples: {results.size}")
    print(f"Mean:    {results.mean():.4f}")
    print(f"Median:  {np.median(results):.4f}")
    print(f"Std:     {results.std(ddof=1):.4f}")
    for q in [5, 25, 50, 75, 95]:
        print(f"p{q:02d}:   {np.percentile(results, q):.4f}")
    print(f"Results: {results}")

    # You can also save `results` for plotting a histogram later.

if __name__ == "__main__":
    main()
