import torch

# once at top-level (optional, can speed up on multi-core CPUs)
torch.set_num_threads(12)          # tune for your CPU
torch.set_num_interop_threads(1)  # often helps
torch.set_grad_enabled(False)     # we’re not training

import numpy as np 
import pandas as pd 
import zarr
from typing import Dict
from gerry_environment import FrankenData, FrankenmanderingEnv  # your classes
from opinion_distribution_utils import zarr_open_experiment_store  # for path helpers
from opinion_distribution import labels_to_action,drf_f4, drf_f1
import pyarrow.parquet as pq
import fastparquet

# ---- CONFIG: point to the experiment + run you want to replay ----
# If DRF_f1 then use these paths
# EXP_PATH = r"F:\Carleton University\Prof Zinovi RA\Code\artifacts_zarr_drf_1\exp-K6_H8xW9_F30_M30_S100_drf_f1_mad.zarr"
# RUN_NAME = "run-20251008-093608-b03dc421"   # or "run-20251008-..." if you used time-based ids
# DRF_FN = drf_f1

# If DRF_f4 then use these paths
EXP_PATH = r"F:\Carleton University\Prof Zinovi RA\Code\artifacts_zarr_drf_4\exp-K6_H8xW9_F30_M30_S100_drf_f4_mad.zarr"
RUN_NAME = "run-20251008-045422-99883f15"   # or "run-20251008-..." if you used time-based ids
DRF_FN = drf_f4

Beta1 = 0.005   # change freely
Beta2 = 0.01   # change freely

# ---- helpers ----
def _np(a): 
    return np.asarray(a)

def rebuild_fd_from_world_init(g_init) -> FrankenData:
    opinion0   = _np(g_init["opinion0"][:]).astype(np.float32)        # (N,m)
    labels0    = _np(g_init["labels0"][:]).astype(np.int64)           # (N,)
    geo_edge   = _np(g_init["geo_edge"][:]).astype(np.int64)          # (2,E_geo)
    social_edge= _np(g_init["social_edge"][:]).astype(np.int64)       # (2,E_soc)
    pos_xy     = _np(g_init["pos_xy"][:]).astype(np.float32)          # (N,2)

    # optional attrs (may or may not exist)
    geo_edge_attr    = g_init.get("geo_edge_attr")
    social_edge_attr = g_init.get("social_edge_attr")
    reps_obj         = g_init.get("reps") or g_init.get("representatives")
    reps             = _np(reps_obj[:]).astype(np.float32) if reps_obj is not None else None
    if geo_edge_attr is not None:    geo_edge_attr    = _np(geo_edge_attr[:]).astype(np.float32)
    if social_edge_attr is not None: social_edge_attr = _np(social_edge_attr[:]).astype(np.float32)
    else:                            social_edge_attr = np.ones(social_edge.shape[1], dtype=np.float32)

    orig_edge_num = social_edge.shape[1]

    # Build FrankenData (matches your __init__ signature)
    fd0 = FrankenData(
        social_edge       = social_edge,
        geographical_edge = geo_edge,
        orig_edge_num     = orig_edge_num,
        opinion           = opinion0,
        pos               = pos_xy,
        reps              = reps,
        dist_label        = labels0,
        edge_attr         = social_edge_attr,
        geo_edge_attr     = (geo_edge_attr if geo_edge_attr is not None else np.ones((geo_edge.shape[1],1), np.float32)),
    )
    return fd0

# --- FAST metrics: create once per (f,m,c); reuse buffers to avoid allocs ---
def make_metric_fn():
    @torch.no_grad()
    def step_metrics_fast(X, C):
        if not torch.is_tensor(X): X = torch.as_tensor(X, dtype=torch.float32)
        else:                      X = X.to(torch.float32)
        if not torch.is_tensor(C): C = torch.as_tensor(C, dtype=torch.float32)
        else:                      C = C.to(torch.float32)

        mean_op     = X.mean().item()
        c_star_mean = C.mean().item()
        shifted     = abs(mean_op - c_star_mean)

        per_node = (X - C).abs().mean(dim=1)  # (N,)
        # If you want L2 (Euclidean) distance instead of MAD, use this instead:
        # L2 per-node distance to c*
        # per_node = torch.linalg.norm(X - C, axis=1).mean(dim=1)  # (N,)
        
        mad = per_node.mean().item()
        sd  = per_node.std(unbiased=False).item()
        return mean_op, mad, sd, shifted, c_star_mean
    return step_metrics_fast

# ---- open run ----
root = zarr.open_group(EXP_PATH, mode="r")
g_run = root["runs"][RUN_NAME]

# try to read c* list for labeling
c_star_list = list(map(float, root.attrs.get("c_star_list", [])))

# ------------- PASS 1: enumerate all (f,m,c) upfront -------------
F_keys = sorted([k for k in g_run.keys() if k.startswith("f_")])
triples = []
for f_key in F_keys:
    g_f = g_run[f_key]
    M_keys = sorted([k for k in g_f.keys() if k.startswith("m_")])
    for m_key in M_keys:
        g_m = g_f[m_key]
        C_keys = sorted([k for k in g_m.keys() if k.startswith("c_")])
        for c_key in C_keys:
            triples.append((f_key, m_key, c_key))

STEPS = 100
ROWS = len(triples) * (STEPS + 1)  # +1 for t=0 snapshot
print("total rows:", ROWS)

# ------------- PREALLOC columns (lean dtypes) -------------
run_col          = np.empty(ROWS, dtype=object)     # constant but keep generic
f_id_col         = np.empty(ROWS, dtype=np.int16)
m_id_col         = np.empty(ROWS, dtype=np.int16)
c_idx_col        = np.empty(ROWS, dtype=np.int8)
step_col         = np.empty(ROWS, dtype=np.int16)
mean_opinion_col = np.empty(ROWS, dtype=np.float32)
mad_col          = np.empty(ROWS, dtype=np.float32)
sd_col           = np.empty(ROWS, dtype=np.float32)
shifted_col      = np.empty(ROWS, dtype=np.float32)
cmean_col        = np.empty(ROWS, dtype=np.float32)

# If RUN_NAME is constant across all rows, fill once:
run_col[:] = RUN_NAME

# ------------- PASS 2: rebuild and fill by index -------------
idx = 0
for (f_key, m_key, c_key) in triples:
    g_c = g_run[f_key][m_key][c_key]
    g_init = g_c["world_init"]

    # rebuild initial state
    fd0 = rebuild_fd_from_world_init(g_init)
    c_star = _np(g_init["c_star"][:]).astype(np.float32)   # (N,m)

    labels0_np = np.asarray(fd0.dist_label, dtype=np.int64).ravel()
    num_districts = int(labels0_np.max()) + 1
    action = labels_to_action(labels0_np, num_districts)

    env = FrankenmanderingEnv(
        num_voters     = fd0.opinion.shape[0],
        num_districts  = num_districts,
        opinion_dim    = fd0.opinion.shape[1],
        horizon        = 100,
        seed           = 0,
        FrankenData    = fd0,
        target_opinion = c_star
    )
    fd, _ = env.reset()

    # set up fast metrics closure once per (f,m,c)
    step_metrics_fast = make_metric_fn()
    c_star = torch.from_numpy(np.asarray(g_init["c_star"][:], dtype=np.float32))

    # parse ids once
    f_id = int(f_key.split("_")[1])
    m_id = int(m_key.split("_")[1])
    c_id = int(c_key.split("_")[1])

    # ---- t = 0 snapshot ----
    mean_op, mad, sd, shifted, c_mean = step_metrics_fast(fd.opinion, c_star)

    run_col[idx]          = RUN_NAME
    f_id_col[idx]         = f_id
    m_id_col[idx]         = m_id
    c_idx_col[idx]        = c_id
    step_col[idx]         = 0
    mean_opinion_col[idx] = mean_op
    mad_col[idx]          = mad
    sd_col[idx]           = sd
    shifted_col[idx]      = shifted
    cmean_col[idx]        = c_mean
    idx += 1

    # ---- t = 1..100 ----
    for t in range(1, 101):
        fd, reward, terminated, truncated, info = env.step(action, DRF_FN, Beta1, Beta2)
        mean_op, mad, sd, shifted, c_mean = step_metrics_fast(fd.opinion, c_star)

        f_id_col[idx]         = f_id
        m_id_col[idx]         = m_id
        c_idx_col[idx]        = c_id
        step_col[idx]         = t
        mean_opinion_col[idx] = mean_op
        mad_col[idx]          = mad
        sd_col[idx]           = sd
        shifted_col[idx]      = shifted
        cmean_col[idx]        = c_mean
        idx += 1

# ------------- Build the DataFrame ONCE -------------
df = pd.DataFrame({
    "run": run_col,
    "f_id": f_id_col,
    "m_id": m_id_col,
    "c_idx": c_idx_col,
    "step": step_col,
    "mean_opinion": mean_opinion_col,
    "mad": mad_col,
    "sd": sd_col,
    "shifted_mean": shifted_col,
    "c_mean": cmean_col,
})

# save CSV
df = df.sort_values(["f_id","m_id","c_idx","step"]).reset_index(drop=True)

# If using drf_f1 then use this path
# print("saving CSV...")
# df.to_csv(f"F:/Carleton University/Prof Zinovi RA/Code/artifacts_zarr_drf_1/exp-K6_H8xW9_F30_M30_S100_drf_f1_mad.zarr/runs/run-20251008-093608-b03dc421/csv_and_histogram/{RUN_NAME}_perstep_metrics.csv", index=False)

# If using drf_f4 then use this path
print("saving Parquet...")
df.to_parquet(f"F:/Carleton University/Prof Zinovi RA/Code/artifacts_zarr_drf_4/exp-K6_H8xW9_F30_M30_S100_drf_f4_mad.zarr/runs/run-20251008-045422-99883f15/csv_and_histogram/{RUN_NAME}_perstep_metrics", index=False)
print(df.shape)  # expect (272700, 9)
