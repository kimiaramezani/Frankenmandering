# experiment_static_franken.py

import json
import numpy as np
import torch
import datetime
from typing import List, Tuple
from gerry_environment import FrankenmanderingEnv
from graph_initiator import build_init_data
from init_graph_to_frankendata import graph_to_frankendata
from opinion_distribution_utils import (zarr_open_experiment_store, zarr_new_run_in_experiment, zarr_world_group, zarr_map_group, zarr_save_world_init,
    zarr_save_step_from_env, zarr_write_summary, zarr_cstar_group)

# from mcmc_baseline import labels_to_action, proposal_flip,

# -------------------- knobs --------------------
K = 6                    # districts per world f
H, W = 8, 9              # grid size for geography (N = H*W)
F = 5                   # number of worlds to sample
M_per_f = 3             # number of random districtings per world
STEPS = 100              # env steps per (f, m)
c_star_list = [0.0,3.5,7.0]   # your custom ideal (all static as defined here in R^m)

# ---- Deterministic seed schedules ----
F_SEEDS = np.arange(1000, 1000 + F, dtype=int)     # world seeds: 1000..1000+F-1
M_BASE  = 5000 

# GEO & SOCIAL configs
NEIGHBORHOOD = "rook"    # or "queen"
BA_m = 2                 # Barabási–Albert parameter
HBO_ALPHA, HBO_BETA = 2.0, 2.0
HBO_INFL = 0.8
SCALE_OUT = 7.0          # opinions in 0..7 (matches your converter default)
MIN_MANHATTAN = 3        # spacing constraint for seed selection

Beta1 = 0.1
Beta2 = 0.5

# DRF hyperparameters (example; tune as needed)
def drf_fig1(discrepancy):
    delta = abs(discrepancy)
 
    if 0 <= delta <=1 :
        return 0  # indifference
 
    elif 1 < delta <= 2:
        return delta-1 # assimilation (y = x-1)
 
    elif 2 < delta <= 3:
        return 1 # assimilation (y = 1)
 
    elif 3 < delta <= 3.2:
        return -2*delta + 7 # assimilation (y=−2x+7)
 
    elif 3.2 < delta < 4:
        return 0  # ambivalence
 
    elif 4 <= delta < 5:
        return -1  # backfire
 
    elif 5 <= delta < 6:
        return  delta - 6 # backfire
 
    elif 6 <= delta  :
        return 0  # irrelevance (ignored)

def drf_fig4(discrepancy):
    delta = abs(discrepancy)
 
    if 0 <= delta < 2:
        return 0  # indifference
 
    elif 2 <= delta < 4:
        return 1  # assimilation (pull closer)
 
    elif 4 <= delta < 6:
        return -1  # backfire (push away)
 
    elif 6 <= delta  :
        return 0  # irrelevance (ignored)
 
    elif delta <= 2:
        return 0  # ambivalence

# eps_indiff, eps_assim, eps_backfire, eps_irrel, eps_amb = 0, 3.0, 3.0, 200.0, 0.0
# assim_shift, back_shift, indiff_shift, amb_shift, irr_shift = 1, -1, 0.0, 0.0, 0.0

# RNG = np.random.default_rng(1234)

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
    Deterministic world f = (geography + opinions + social).
    """
    fd, G = build_init_data(
        K=K, H=H, W=W, ba_m=BA_m, rng_seed=seed, # <— world seed drives BOTH social & opinions
        neighborhood=NEIGHBORHOOD,
        hbo_alpha=HBO_ALPHA, hbo_beta=HBO_BETA, hbo_influence=HBO_INFL,
        scale_out=SCALE_OUT, min_manhattan=MIN_MANHATTAN,
        attach_hetero=False, use_scaled_opinion=True
    )
    return fd, G

def random_maps_for_world(G, K: int, m_count: int, base_seed: int) -> List[np.ndarray]:
    """
    Generate m_count random district labelings for the same world f
    by reseeding the spaced-seed picker + greedy fill each time.
    Deterministic maps for this world: seed_j = base_seed + j.
    """
    labels_list = []
    for j in range(m_count):
        s = base_seed + j
        seeds = G.choose_random_district_seeds_spaced(K, min_manhattan=MIN_MANHATTAN, rng_seed=s)
        labels = G.greedy_fill_districts(seeds, rng_seed=s)
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

# We can delete this function as it is not used.
def op_diff(fd, K: int, steps: int, drf,Beta1,Beta2) -> float:
    """
    Run `steps` with a static (hard) assignment equal to fd.dist_label.
    Collect final distance-to-ideal (sum of MAD to c*).
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
    # Initial MAD distance to c* (per voter average) measured before applying any step
    init_dist = float(np.mean(np.abs(np.squeeze(x_t) - np.squeeze(c_star))))
    print(f"Initial Distance:t= 0 (pre-step)  sum|x|={init_dist:.3f}")

    for t in range(steps):
        obs, reward, terminated, truncated, info = env.step(action, drf_fig4, Beta1, Beta2)

        # ---- DEBUG PEEK: after this step ----
        # choose any checkpoints you want; these hit early/mid/last
        if t in (0, steps//2 - 1, steps - 1):
            x_t = np.asarray(obs.opinion)
            step_mean = float(np.mean(np.abs(np.squeeze(x_t) - np.squeeze(c_star))))
            print(f"t={t+1:3d} (post-step) sum|x|={step_mean:.3f}")

        if terminated or truncated:
            break

    x_final = np.asarray(obs.opinion)
    # print(f"x_final mean: {x_final.mean()} ...")
    # final distance to c*
    final_dist = float(np.mean(np.abs(np.squeeze(x_final) - np.squeeze(c_star))))
    print(f"Final distance:", final_dist)
    print(f"Final distance = Step Mean 100:", final_dist == step_mean)
    opinion_dist_change = final_dist - init_dist
    print(f"opinion_dist_change = Final - Initial:", opinion_dist_change)

    # Histogram of the averages across runs
    return float(final_dist)

def exp_slug(K,H,W,F,M_per_f,STEPS, drf_name="4", metric="mad"):
    return f"exp-K{K}_H{H}xW{W}_F{F}_M{M_per_f}_S{STEPS}_{drf_name}_{metric}"

def main():
    # one Zarr run per experiment (keeps previous runs intact)
    # (choose a root dir; or define ZARR_ROOT in utils and omit root_dir=)
    # 1) OPEN (or create) the single experiment store
    EXPERIMENT = exp_slug(K,H,W,F,M_per_f,STEPS, drf_name="drf_fig4", metric="mad")
    root_exp, EXP_PATH = zarr_open_experiment_store(
        root_dir="artifacts_zarr",
        experiment_slug=EXPERIMENT,
        overwrite=False,
        attrs={"experiment": EXPERIMENT,
               "K": int(K), "H": int(H), "W": int(W),
               "F": int(F), "M_per_f": int(M_per_f),
               "STEPS": int(STEPS),
               "c_star_list": list(map(float, c_star_list))}
    )

    # 2) CREATE a new run subgroup under /runs
    #    Use run_index if you want 000, 001, ...; or drop it to get time-based run ids.
    g_run, RUN_NAME = zarr_new_run_in_experiment(
        root_exp,
        run_index=None,   # e.g., set to 34 to force /runs/run_034
        attrs={"created_at": datetime.datetime.now().isoformat()}
    )
    print(f"[zarr] writing to {EXP_PATH} :: /runs/{RUN_NAME}")

    # aggregate metrics across all (f, m, c)
    finals = []
    deltas = []
    index  = []  # (f_id, m_id, c_idx)

    for f_id, f_seed in enumerate(F_SEEDS):
        fd_world, G = sample_world(K, H, W, f_seed)
        g_f = zarr_world_group(g_run, f_id, int(f_seed))

        M_SEED_BASE = M_BASE + f_id * 10
        labels_list = random_maps_for_world(G, K, M_per_f, base_seed=M_SEED_BASE)

        pos_xy_init   = G.df_nodes[["x","y"]].to_numpy(np.float32)
        node_ids_init = G.df_nodes["id"].to_numpy(np.int32)

        for m_id, labels in enumerate(labels_list):
            fd = fd_with_labels(G, K, labels)
            g_m = zarr_map_group(g_f, m_id, int(M_SEED_BASE + m_id))

            N, m = fd.opinion.shape
            geo_edge    = np.asarray(fd.geographical_edge)
            social_edge = np.asarray(fd.social_edge)
            geo_attr    = getattr(fd, "geo_edge_attr", None)
            soc_attr    = getattr(fd, "social_edge_attr", None)

            for c_idx, c_val in enumerate(c_star_list):
                c_star = np.full((N, m), float(c_val), dtype=np.float32)
                g_c = zarr_cstar_group(g_m, c_idx, c_val)

                x0 = np.asarray(fd.opinion)
                init_dist = float(np.mean(np.abs(np.squeeze(x0) - np.squeeze(c_star))))
                zarr_save_world_init(g_c, fd=fd, G=G, c_star=c_star, init_dist=init_dist, compressor=None)

                env = FrankenmanderingEnv(
                    num_voters=N, num_districts=K, opinion_dim=m,
                    horizon=STEPS, seed=0,
                    FrankenData=fd, target_opinion=c_star,
                )
                action = labels_to_action(np.asarray(fd.dist_label, dtype=np.int32), K)
                obs, info = env.reset()

                CHECK_T = {1,25,50,75,100}
                last_mean = None

                for t in range(STEPS):
                    obs, reward, terminated, truncated, info = env.step(action, drf_fig4, Beta1, Beta2)
                    t_post = t + 1
                    if t_post in CHECK_T:
                        x_t = np.asarray(obs.opinion)
                        mean_mad = float(np.mean(np.abs(np.squeeze(x_t) - np.squeeze(c_star))))
                        last_mean = mean_mad
                        zarr_save_step_from_env(
                            g_c, t_post, env=env, c_star=c_star, mean_mad_to_cstar=mean_mad, compressor=None,
                            geo_edge_override=geo_edge, social_edge_override=social_edge,
                            geo_edge_attr_override=geo_attr, social_edge_attr_override=soc_attr,
                            pos_xy_override=pos_xy_init, node_ids_override=node_ids_init,
                            representatives_override=getattr(env, "last_representatives", None),
                        )
                    if terminated or truncated:
                        break

                x_final = np.asarray(obs.opinion)
                final_dist = float(np.mean(np.abs(np.squeeze(x_final) - np.squeeze(c_star))))
                delta = final_dist - init_dist

                zarr_write_summary(g_c, final_dist, last_checkpoint_mean=last_mean)
                g_c.require_group("summary").attrs["delta_mean_mad"] = float(delta)
                g_c.require_group("summary").attrs["init_mean_mad_copy"] = float(init_dist)

                finals.append(final_dist); deltas.append(delta); index.append((f_id, m_id, c_idx))

    finals = np.asarray(finals, dtype=np.float64)
    deltas = np.asarray(deltas, dtype=np.float64)
    print(f"[done] Zarr written to: {EXP_PATH} :: /runs/{RUN_NAME}")
    print(f"Final mean={finals.mean():.4f}, median={np.median(finals):.4f}")
    print(f"Delta  mean={deltas.mean():.4f}, median={np.median(deltas):.4f}")

if __name__ == "__main__":
    main()
