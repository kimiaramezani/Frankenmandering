# ---opinion_distribution_utils.py

import os, uuid, datetime
import numpy as np
import zarr
from numcodecs import Blosc
import json

# -------------------- Utilities --------------------

ZARR_ROOT = "artifacts_zarr"

# Version-agnostic compressor: v3 -> zarr.codecs.Blosc, v2 -> numcodecs.Blosc
try:
    # Zarr v3 path
    from zarr.codecs import Blosc as ZarrBlosc
    _COMPRESSOR = ZarrBlosc(cname="zstd", clevel=7, shuffle=ZarrBlosc.SHUFFLE)
    _USE_V3_COMPRESSORS = True
except Exception:
    # Zarr v2 fallback
    from numcodecs import Blosc as NumBlosc
    _COMPRESSOR = NumBlosc(cname="zstd", clevel=7, shuffle=NumBlosc.SHUFFLE)
    _USE_V3_COMPRESSORS = False

# --- v2/v3-agnostic dataset creator with NO compression (keeps things simple) ---
def _compress_kwargs(_comp):
    return {}  # no compression; avoids v3 codec differences

def _create_ds(g, name, data, chunks):
    arr = np.asarray(data)
    return g.create_dataset(
        name,
        data=arr,
        shape=arr.shape,   # required by Zarr v3
        chunks=chunks,
        overwrite=True,    # okay in both v2/v3
        **_compress_kwargs(None),
    )

def _np(x):
    """Torch->NumPy safe cast; otherwise np.asarray."""
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)

def zarr_open_experiment_store(root_dir=ZARR_ROOT, experiment_slug="exp-default", overwrite=False, attrs=None):
    """
    Open a SINGLE store for the whole experiment at:
      artifacts_zarr/<experiment_slug>.zarr
    Returns (root_exp, exp_path).
    """
    os.makedirs(root_dir, exist_ok=True)
    exp_path = os.path.join(root_dir, f"{experiment_slug}.zarr")
    mode = 'w' if overwrite else 'a'
    root_exp = zarr.open_group(exp_path, mode=mode)
    if attrs:
        root_exp.attrs.update(attrs)
    # ensure 'runs' container exists
    root_exp.require_group("runs")
    return root_exp, exp_path

def zarr_new_run_in_experiment(root_exp, run_index=None, run_id=None, attrs=None):
    """
    Create a new subgroup under /runs for this execution:
      /runs/run_<index:03d>  or  /runs/run-<run_id>
    Returns (g_run, run_name).
    """
    runs = root_exp["runs"]

    if run_index is not None:
        name = f"run_{int(run_index):03d}"
        if name in runs:
            # open existing to append (or raise if you prefer)
            g_run = runs[name]
        else:
            g_run = runs.create_group(name)
    else:
        if run_id is None:
            run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
        name = f"run-{run_id}"
        g_run = runs.create_group(name)

    if attrs:
        g_run.attrs.update(attrs)
    return g_run, name

def zarr_new_run_store(root_dir=ZARR_ROOT, run_id=None, overwrite=False, attrs=None):
    """
    Create a new Zarr DirectoryStore for this experiment run:
      artifacts_zarr/run-<run_id>.zarr
    """
    if run_id is None:
        run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
    run_path = os.path.join(root_dir, f"run-{run_id}.zarr")
    os.makedirs(root_dir, exist_ok=True)  # ensure parent exists

    # mode='w' for fresh run, 'w-' to error if exists, 'a' to append
    mode = 'w' if overwrite else 'w-'
    try:
        root = zarr.open_group(run_path, mode=mode)
    except Exception:
        # fallback: if 'w-' failed because it exists, open in 'a'
        root = zarr.open_group(run_path, mode='a')

    if attrs:
        root.attrs.update(attrs)
    # return the correct compressor object for this runtime
    return root, _COMPRESSOR, run_path, run_id

def zarr_world_group(root, f_id, f_seed):
    g_f = root.require_group(f"f_{f_id:03d}")
    g_f.attrs["f_seed"] = int(f_seed)
    return g_f

def zarr_map_group(g_f, m_id, m_seed):
    g_m = g_f.require_group(f"m_{m_id:03d}")
    g_m.attrs["m_seed"] = int(m_seed)
    return g_m

def zarr_save_world_init(g_m,*,
    fd,            # FrankenData @ t=0
    G,             # your Grid-like object (for pos_xy, node_ids)
    c_star,        # (N,m) target (from driver)
    init_dist,     # scalar mean mad to c_star (from driver)
    compressor: Blosc):
    """
    Save t=0 inputs for this (f,m): opinions, labels, geo/social edges, positions, c_star.
    Also store init_dist as group attrs.
    """
    g_init = g_m.require_group("world_init")

    opinion0   = _np(fd.opinion).astype(np.float32)            # (N,m)
    labels0    = _np(fd.dist_label).astype(np.int32)           # (N,)
    geo_edge   = _np(fd.geographical_edge).astype(np.int32)    # (2,E_geo)
    social_edge= _np(fd.social_edge).astype(np.int32)          # (2,E_soc)

    # Optional attributes if present on fd
    geo_edge_attr    = getattr(fd, "geo_edge_attr", None)
    social_edge_attr = getattr(fd, "social_edge_attr", None)
    if geo_edge_attr is not None:
        geo_edge_attr = _np(geo_edge_attr).astype(np.float32)
    if social_edge_attr is not None:
        social_edge_attr = _np(social_edge_attr).astype(np.float32)

    pos_xy   = G.df_nodes[["x","y"]].to_numpy(np.float32)
    node_ids = G.df_nodes["id"].to_numpy(np.int32)
    c_star   = _np(c_star).astype(np.float32)

    N, m = opinion0.shape
    _create_ds(g_init, "opinion0",  data=opinion0,  chunks=(min(N,8192), m))
    _create_ds(g_init,"labels0",   data=labels0,   chunks=(min(N,8192),))
    _create_ds(g_init,"geo_edge",  data=geo_edge,  chunks=(2, min(geo_edge.shape[1], 65536)))
    _create_ds(g_init,"social_edge", data=social_edge, chunks=(2, min(social_edge.shape[1], 65536)))

    if geo_edge_attr is not None:
        _create_ds(g_init,"geo_edge_attr", data=geo_edge_attr,
                              chunks=(min(geo_edge_attr.shape[0], 65536), geo_edge_attr.shape[1]))
    if social_edge_attr is not None:
        _create_ds(g_init,"social_edge_attr", data=social_edge_attr,
                              chunks=(min(social_edge_attr.shape[0], 65536), social_edge_attr.shape[1]))

    _create_ds(g_init,"pos_xy",   data=pos_xy,   chunks=(min(N,8192), 2))
    _create_ds(g_init,"node_ids", data=node_ids, chunks=(min(N,8192),))
    _create_ds(g_init,"c_star",   data=c_star,   chunks=(min(N,8192), m))

    # distance is PASSED IN (no internal math)
    g_init.attrs["init_dist_mean_mad"] = float(init_dist)

def zarr_save_step_from_env(
    g_m,
    t: int,
    *,
    env,                   # environment after step t (holds FrankenData @ t)
    c_star,                # (N,m) target (from driver)
    mean_mad_to_cstar,      # scalar mean MAD to c_star (from driver)
    compressor: Blosc,
    # Optional overrides if you later change edges/attrs during the run.
    geo_edge_override=None,
    social_edge_override=None,
    geo_edge_attr_override=None,
    social_edge_attr_override=None,
    pos_xy_override=None,
    node_ids_override=None,
    representatives_override=None):
    """Save full snapshot for step t. Pulls arrays from env.FrankenData unless overrides are supplied."""
    fd = env.FrankenData

    opinion = _np(fd.opinion).astype(np.float32)
    labels  = _np(fd.dist_label).astype(np.int32)
    c_star  = _np(c_star).astype(np.float32)

    # Use overrides if provided, else what’s in fd / last known static versions
    geo_edge  = _np(geo_edge_override).astype(np.int32)  if geo_edge_override  is not None else _np(fd.geographical_edge).astype(np.int32)
    social_edge = _np(social_edge_override).astype(np.int32) if social_edge_override is not None else _np(fd.social_edge).astype(np.int32)

    geo_edge_attr    = _np(geo_edge_attr_override).astype(np.float32)    if geo_edge_attr_override    is not None else getattr(fd, "geo_edge_attr", None)
    if geo_edge_attr is not None:
        geo_edge_attr = _np(geo_edge_attr).astype(np.float32)

    social_edge_attr = _np(social_edge_attr_override).astype(np.float32) if social_edge_attr_override is not None else getattr(fd, "social_edge_attr", None)
    if social_edge_attr is not None:
        social_edge_attr = _np(social_edge_attr).astype(np.float32)

    # positions & ids: allow override (e.g., if you move nodes later)
    if pos_xy_override is not None:
        pos_xy = _np(pos_xy_override).astype(np.float32)
    else:
        # if not passed, try to read from env/frankendata (if stored) or keep from G at t=0;
        # since fd typically doesn’t carry pos, we expect you to reuse the t=0 values as override if you want parity every step
        pos_xy = None

    if node_ids_override is not None:
        node_ids = _np(node_ids_override).astype(np.int32)
    else:
        node_ids = None

    reps = representatives_override
    if reps is None:
        reps = getattr(env, "last_representatives", None)
    if reps is not None:
        reps = _np(reps).astype(np.int32)

    g_steps = g_m.require_group("steps")
    g_t = g_steps.require_group(f"t_{int(t):04d}")

    N, m = opinion.shape
    _create_ds(g_t,"opinion", data=opinion, chunks=(min(N,8192), m))
    _create_ds(g_t,"labels",  data=labels,  chunks=(min(N,8192),))
    _create_ds(g_t,"c_star",  data=c_star,  chunks=(min(N,8192), m))
    _create_ds(g_t,"geo_edge", data=geo_edge, chunks=(2, min(geo_edge.shape[1], 65536)))
    _create_ds(g_t,"social_edge", data=social_edge, chunks=(2, min(social_edge.shape[1], 65536)))

    if geo_edge_attr is not None:
        _create_ds(g_t,"geo_edge_attr", data=geo_edge_attr,
                   chunks=(min(geo_edge_attr.shape[0], 65536), geo_edge_attr.shape[1]))
    if social_edge_attr is not None:
        _create_ds(g_t,"social_edge_attr", data=social_edge_attr,
                   chunks=(min(social_edge_attr.shape[0], 65536), social_edge_attr.shape[1]),)

    if pos_xy is not None:
        _create_ds(g_t,"pos_xy",    data=pos_xy,   chunks=(min(N,8192), 2),)
    if node_ids is not None:
        _create_ds(g_t,"node_ids",  data=node_ids, chunks=(min(N,8192),))
    if reps is not None:
        _create_ds(g_t,"representatives", data=reps, chunks=(min(reps.size,8192),))

    # distance is PASSED IN (no internal math)
    g_t.attrs["mean_mad_to_cstar"] = float(mean_mad_to_cstar)

def zarr_write_summary(g_m, final_dist, last_checkpoint_mean=None):
    g_sum = g_m.require_group("summary")
    g_sum.attrs["final_mean_mad_to_cstar"] = float(final_dist)
    if last_checkpoint_mean is not None:
        g_sum.attrs["last_checkpoint_mean"] = float(last_checkpoint_mean)
        
def zarr_cstar_group(g_m, c_idx: int, c_value):
    """
    Create f_###/m_###/c_### group and tag its attrs with c_star meta.
    c_value can be a scalar or list/array; we store both a scalar preview and raw JSON if needed.
    """
    g_c = g_m.require_group(f"c_{int(c_idx):03d}")
    # best-effort metadata
    try:
        # If scalar-like, cast to float; else store None
        g_c.attrs["c_star_scalar"] = float(c_value) if np.isscalar(c_value) else None
    except Exception:
        g_c.attrs["c_star_scalar"] = None
    # Also store a JSON string for exact replay (scalar or list)
    try:
        g_c.attrs["c_star_json"] = json.dumps(c_value if np.isscalar(c_value) else np.asarray(c_value).tolist())
    except Exception:
        g_c.attrs["c_star_json"] = None
    return g_c
