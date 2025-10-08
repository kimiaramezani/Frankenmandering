import numpy as np, pandas as pd, zarr
from typing import Dict
from gerry_environment import FrankenData, FrankenmanderingEnv  # your classes
from opinion_distribution_utils import zarr_open_experiment_store  # for path helpers
from opinion_distribution import labels_to_action,drf_f4, drf_f1

# ---- CONFIG: point to the experiment + run you want to replay ----
# If DRF_f1 then use these paths
EXP_PATH = r"F:\Carleton University\Prof Zinovi RA\Code\artifacts_zarr_drf_1\exp-K6_H8xW9_F30_M30_S100_drf_f1_mad.zarr"
RUN_NAME = "run-20251008-093608-b03dc421"   # or "run-20251008-..." if you used time-based ids
DRF_FN = drf_f1

# If DRF_f4 then use these paths
# EXP_PATH = r"F:\Carleton University\Prof Zinovi RA\Code\artifacts_zarr_drf_4\exp-K6_H8xW9_F30_M30_S100_drf_f4_mad.zarr"
# RUN_NAME = "run-20251008-045422-99883f15"   # or "run-20251008-..." if you used time-based ids
# DRF_FN = drf_f4

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
        reps              = None,
        dist_label        = labels0,
        edge_attr         = social_edge_attr,
        geo_edge_attr     = (geo_edge_attr if geo_edge_attr is not None else np.ones((geo_edge.shape[1],1), np.float32)),
    )
    return fd0

def mean_abs_stats(opinion: np.ndarray, c_star: np.ndarray) -> Dict[str, float]:
    # opinion, c_star both (N,m); distances in same units you’re using now
    d = np.abs(opinion - c_star)
    # if m>1 this is elementwise abs — which matches your “MAD to c*” definition in the run;
    # if you instead want L2-to-c* then replace with: d = np.linalg.norm(opinion - c_star, axis=1, keepdims=True)
    # and drop keepdims below.
    d = d.reshape(d.shape[0], -1).mean(axis=1, keepdims=True)  # average over m dims → per-node scalar
    return {
        "mad": float(d.mean()),
        "sd":  float(d.std(ddof=0)),
        "mean_opinion": float(opinion.reshape(opinion.shape[0], -1).mean())
    }

def step_metrics(opinion: np.ndarray, c_star: np.ndarray) -> Dict[str, float]:
    X = np.asarray(opinion, dtype=np.float32)
    C = np.asarray(c_star, dtype=np.float32)

    mean_opinion_t = float(X.mean())
    c_mean = float(C.mean())
    shifted_mean_t = abs(mean_opinion_t - c_mean)

    per_node_avg_abs = np.abs(X - C).mean(axis=1)  # (N,)
    mad_t = float(per_node_avg_abs.mean())
    sd_t  = float(per_node_avg_abs.std(ddof=0))

    return {
        "mad": mad_t,
        "sd": sd_t,
        "mean_opinion": mean_opinion_t,
        "shifted_mean": shifted_mean_t,
        "c_mean": c_mean,
    }

# ---- open run ----
root = zarr.open_group(EXP_PATH, mode="r")
g_run = root["runs"][RUN_NAME]

# try to read c* list for labeling
c_star_list = list(map(float, root.attrs.get("c_star_list", [])))

rows = []
for f_key in sorted([k for k in g_run.keys() if k.startswith("f_")]):
    g_f = g_run[f_key]
    for m_key in sorted([k for k in g_f.keys() if k.startswith("m_")]):
        g_m = g_f[m_key]

        # Each (f,m) has 3 c_* groups (c_000, c_001, c_002). We will rebuild from world_init of each c.
        for c_key in sorted([k for k in g_m.keys() if k.startswith("c_")]):
            g_c = g_m[c_key]
            g_init = g_c["world_init"]
            fd0 = rebuild_fd_from_world_init(g_init)
            c_star = _np(g_init["c_star"][:]).astype(np.float32)   # (N,m)
            labels0_np = np.asarray(fd0.dist_label, dtype=np.int64).ravel()
            num_districts = int(labels0_np.max()) + 1
            action = labels_to_action(labels0_np, num_districts)

            # env with fixed horizon=100, same DRF and betas you used originally
            # If your step signature requires DRF/Beta1/Beta2, set them here:
            env = FrankenmanderingEnv(
                num_voters = fd0.opinion.shape[0],
                num_districts = num_districts,
                opinion_dim = fd0.opinion.shape[1],
                horizon = 100,
                seed = 0,
                FrankenData = fd0,
                target_opinion = c_star
            )
            fd, _ = env.reset()

            # t=0 metrics (from world_init)
            m0 = step_metrics(fd.opinion, c_star)
            rows.append({"run": RUN_NAME, "f_id": int(f_key.split("_")[1]), "m_id": int(m_key.split("_")[1]),
                        "c_idx": int(c_key.split("_")[1]), "step": 0,
                        "mean_opinion": m0["mean_opinion"], "mad": m0["mad"], "sd": m0["sd"],
                        "shifted_mean": m0["shifted_mean"], "c_mean": m0["c_mean"],
                    })

            # do 100 steps; use a no-op “keep labels” policy or your preferred policy
            # Here we just keep current labels (one-hot) to drive opinion dynamics via reps augmentation.
            for t in range(1, 101):
                labels = fd.dist_label
                # Supply DRF and Betas you used in original run (f1/f4, Beta1, Beta2); example:
                fd, reward, terminated, truncated, info = env.step(action, DRF=DRF_FN, Beta1=Beta1, Beta2=Beta2)

                mt = step_metrics(fd.opinion, c_star)
                rows.append({"run": RUN_NAME, "f_id": int(f_key.split("_")[1]), "m_id": int(m_key.split("_")[1]),
                            "c_idx": int(c_key.split("_")[1]), "step": t,
                            "mean_opinion": mt["mean_opinion"], "mad": mt["mad"], "sd": mt["sd"],
                            "shifted_mean": mt["shifted_mean"], "c_mean": mt["c_mean"],
                        })

# save CSV
df = pd.DataFrame(rows).sort_values(["f_id","m_id","c_idx","step"]).reset_index(drop=True)
# If using drf_f1 then use this path
df.to_csv(f"F:/Carleton University/Prof Zinovi RA/Code/artifacts_zarr_drf_1/exp-K6_H8xW9_F30_M30_S100_drf_f1_mad.zarr/runs/run-20251008-093608-b03dc421/csv_and_histogram/{RUN_NAME}_perstep_metrics.csv", index=False)
# If using drf_f4 then use this path
# df.to_csv(f"F:/Carleton University/Prof Zinovi RA/Code/artifacts_zarr_drf_4/exp-K6_H8xW9_F30_M30_S100_drf_f4_mad.zarr/runs/run-20251008-045422-99883f15/csv_and_histogram/{RUN_NAME}_perstep_metrics.csv", index=False)
print(df.shape)  # expect (272700, 7)
