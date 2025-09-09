# make_grid.py  — geometry-style grid CSVs (x→right, y→up; origin at bottom-left)

import argparse, datetime as dt
import numpy as np
import pandas as pd
from pathlib import Path
import json, os
# YAML is optional; define 'yaml' either way so it's in scope
try:
    import yaml
except Exception:
    yaml = None

# --- config loader ---
def load_config(path):
    ext = os.path.splitext(path)[1].lower()
    with open(path, "r", encoding="utf-8") as f:
        if ext in (".yaml", ".yml"):
            if yaml is None:
                raise RuntimeError("pyyaml not installed; pip install pyyaml")
            return yaml.safe_load(f) or {}
        elif ext == ".json":
            return json.load(f)
        else:
            raise ValueError(f"Unsupported config type: {ext}")

def pick(val, *candidates):
    """Return the first non-None value among [val] + candidates, else None."""
    for v in (val, *candidates):
        if v is not None:
            return v
    return None

# ID Nomenclature from Row-Column for (x,y) in a grid of width W and height H
def id_from_xy(x, y, W): return y * W + x

# Opinion is here a placeholder. It will be replaced by HBO Model
def build_nodes(H, W, population=1, opinion=0.5, area=1.0):
    xs, ys = np.meshgrid(np.arange(W), np.arange(H))   # ys: 0..H-1 (bottom->up)
    # Flatten the grid to 1D array using ravel() so that we can make changes easily.
    # Those changes will be reflected in the original 2D grid.
    x = xs.ravel().astype(np.int16)
    y = ys.ravel().astype(np.int16)
    ids = (y * W + x).astype(np.int32)
    df = pd.DataFrame({
        "id": ids,
        "x": x,
        "y": y,
        "population": np.full_like(ids, population, dtype=np.int32),
        "opinion": np.full(ids.shape, opinion, dtype=np.float32),
        "area": np.full(ids.shape, area, dtype=np.float32),
        "seed_flag": np.zeros_like(ids, dtype=np.int8),
        "group": pd.Series([""] * ids.size, dtype="string")
    })
    # integrity
    N = H * W
    assert len(df) == N and df["id"].is_unique and df["id"].min() == 0 and df["id"].max() == N-1
    return df

def build_edges_grid(H, W):
    rows = []
    # right edges
    for y in range(H):
        for x in range(W-1):
            u = id_from_xy(x,   y, W)
            v = id_from_xy(x+1, y, W)
            rows.append((min(u,v), max(u,v)))
    # up edges
    for y in range(H-1):
        for x in range(W):
            u = id_from_xy(x, y,   W)
            v = id_from_xy(x, y+1, W)
            rows.append((min(u,v), max(u,v)))
    edges = pd.DataFrame(rows, columns=["u","v"]).drop_duplicates()
    edges["weight_grid"] = edges.shape[0] * [1.0]
    edges["barrier_flag"] = edges.shape[0] * [0]
    # integrity
    E_expected = H*(W-1) + W*(H-1)
    assert len(edges) == E_expected
    return edges

def _sanitize_and_backfill(seeds, H, W, K, rng=42):
    # keep in-bounds, unique, preserve order
    clean = [(int(x), int(y)) for (x, y) in seeds if 0 <= int(x) < W and 0 <= int(y) < H]
    clean = list(dict.fromkeys(clean))  # dedupe

    if len(clean) >= K:
        return clean[:K]

    # farthest-point sampling to reach K unique seeds
    rng = np.random.default_rng(rng)
    all_cells = np.array([(x, y) for y in range(H) for x in range(W)], dtype=int)
    if not clean:
        clean.append((int(rng.integers(0, W)), int(rng.integers(0, H))))
    while len(clean) < K:
        P = np.array(clean, dtype=float)
        d2 = np.min(np.sum((all_cells[:, None, :] - P[None, :, :])**2, axis=2), axis=1)
        idx = int(np.argmax(d2))
        cand = tuple(map(int, all_cells[idx]))
        if cand not in clean:
            clean.append(cand)
    return clean

def load_seeds_from_file(path, H, W, K, key, rng=42):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".yaml", ".yml"):
        if yaml is None:
            raise RuntimeError("pyyaml not installed; pip install pyyaml")
        with open(path, "r", encoding="utf-8") as f:
            doc = yaml.safe_load(f)
        if not key or "presets" not in doc:
            raise ValueError("Provide --seeds-key and ensure 'presets' exists in YAML")
        # key format: "<grid_key>:<name>", e.g., "12x12_K6:runA"
        grid_key, name = key.split(":", 1)
        raw = doc["presets"][grid_key][name]
        return _sanitize_and_backfill(raw, H, W, K, rng)
    elif ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            doc = json.load(f)
        grid_key, name = key.split(":", 1)
        raw = doc["presets"][grid_key][name]
        return _sanitize_and_backfill(raw, H, W, K, rng)
    elif ext == ".csv":
        # CSV variant: one file per preset, two columns x,y
        df = pd.read_csv(path)
        raw = list(map(tuple, df[["x","y"]].to_numpy()))
        return _sanitize_and_backfill(raw, H, W, K, rng)
    else:
        raise ValueError(f"Unsupported seeds file type: {ext}")

def choose_seeds(H, W, K, preset=None, rng=42):
    """
    Always return exactly K unique, in-bounds (x,y) seeds.
    If a preset is provided, sanitize + farthest-point backfill to K.
    Otherwise, start from coarse grid candidates and backfill as needed.
    """
    import numpy as np

    if preset is not None:
        return _sanitize_and_backfill(preset, H, W, K, rng)

    # No preset: start from coarse candidates, then backfill if needed
    side = int(np.ceil(np.sqrt(K)))
    xs = np.linspace(0.1, 0.9, min(side, W)) * (W - 1)
    ys = np.linspace(0.1, 0.9, min(side, H)) * (H - 1)
    candidates = np.array([(int(round(x)), int(round(y))) for y in ys for x in xs], dtype=int)

    # dedupe & clip
    candidates = np.unique(candidates, axis=0)
    candidates[:, 0] = np.clip(candidates[:, 0], 0, W - 1)
    candidates[:, 1] = np.clip(candidates[:, 1], 0, H - 1)

    picks = [tuple(map(int, xy)) for xy in candidates[:K]]
    if len(picks) < K:
        picks = _sanitize_and_backfill(picks, H, W, K, rng)
    return picks

def write_partition_init(nodes_df, H, W, K, seeds_xy, base, outdir, plan_name="init", fname_suffix="partition_init"):
    """
    Write __partition_init.csv with exactly K unique seeds labeled 0..K-1.
    Robust to out-of-bounds/duplicate seeds; prints what was used/skipped.
    """
    part = nodes_df[["id"]].copy()
    part["district"] = -1
    # The locked_flag is a freeze switch for a cell (node). 
    # 1 means the cell is locked and cannot be changed during districting.
    # 0 means the cell is not locked and can be changed.
    part["locked_flag"] = 0
    part["plan_name"] = plan_name

    used = []
    skipped = []
    seen_ids = set()

    for d, (x, y) in enumerate(seeds_xy):
        x = int(x); y = int(y)
        # bounds check
        if not (0 <= x < W and 0 <= y < H):
            skipped.append((d, x, y, "out_of_bounds"))
            continue
        nid = id_from_xy(x, y, W)
        if nid in seen_ids:
            skipped.append((d, x, y, "duplicate"))
            continue
        mask = (part["id"] == nid)
        if not mask.any():
            skipped.append((d, x, y, "id_missing"))
            continue

        seen_ids.add(nid)
        used.append((d, x, y, nid))
        part.loc[mask, "district"] = d
        nodes_df.loc[mask, "seed_flag"] = 1

    print(f"[seeds] requested={len(seeds_xy)} unique_used={len(used)} ids={[u[3] for u in used]}")
    if skipped:
        print(f"[seeds] skipped={len(skipped)} details={skipped}")

    assert len(used) == K, f"Expected K={K} unique seeds, got {len(used)}"

    pth = Path(outdir, f"{base}__{fname_suffix}.csv")
    part.to_csv(pth, index=False)
    return pth

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None, help="YAML/JSON config")

    # Make these OPTIONAL so config can supply them
    ap.add_argument("--H", type=int, default=None)
    ap.add_argument("--W", type=int, default=None)
    ap.add_argument("--K", type=int, default=None)

    # Population Toleration Epsilon
    ap.add_argument("--pop_tol_eps", type=float, default=None)

    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--outdir", type=str, default=None)
    ap.add_argument("--base", type=str, default=None, help="Override BASE name")
    ap.add_argument("--write_seeds", action="store_true", help="Also write partition_init.csv")
    ap.add_argument("--seeds-file", type=str, default=None)
    ap.add_argument("--seeds-key",  type=str, default=None)  # e.g. "12x12_K6:runA"
    args = ap.parse_args()

    cfg = load_config(args.config) if args.config else {}

    # Resolve values: CLI overrides config, which overrides defaults
    H     = pick(args.H,     cfg.get("grid", {}).get("height_H"))
    W     = pick(args.W,     cfg.get("grid", {}).get("width_W"))
    K     = pick(args.K,     cfg.get("grid", {}).get("districts_K"))
    pop_tol_eps   = pick(args.pop_tol_eps,   cfg.get("grid", {}).get("pop_tol_eps"), 0.05)
    seed  = pick(args.seed,  cfg.get("determinism", {}).get("rng_seed"), 42)
    outdir_val = pick(args.outdir, cfg.get("io", {}).get("outdir"), ".")
    base_override = pick(args.base, cfg.get("io", {}).get("base"))

    seeds_file = pick(args.seeds_file, cfg.get("seeds", {}).get("file"))
    seeds_key  = pick(args.seeds_key,  cfg.get("seeds", {}).get("key"))

    # Require H/W/K from either CLI or config
    if H is None or W is None or K is None:
        raise SystemExit("Missing H/W/K. Provide via --H/--W/--K or config 'grid:{H,W,K}'.")

    # Convenient default: if presets are in the same config file,
    # use that file as the seeds source automatically.
    if seeds_file is None and "presets" in cfg and args.config:
        seeds_file = args.config

    # (Optional) Friendly error if you asked to write seeds but didn't pick which run
    if args.write_seeds and seeds_key is None and "presets" in cfg:
        # show available "<gridKey>:<runName>" choices
        choices = []
        for gkey, runs in cfg["presets"].items():
            for rname in runs.keys():
                choices.append(f"{gkey}:{rname}")
        raise SystemExit(
            "Missing seeds_key. Set seeds.key in YAML or pass --seeds-key.\n"
            f"Available: {', '.join(choices)}"
        )
    
    # Build BASE using the resolved values (not args.*)
    # before writing files
    now = dt.datetime.now()                # local time; use dt.datetime.utcnow() if you prefer UTC
    date_tag = now.strftime("%y%m%d")      # e.g., 250903 for Sep 3, 2025
    time_tag = now.strftime("%H%M%S")      # e.g., 142357 for 14:23:57
    base = base_override or f"grid_{W}x{H}__K{K}__adj4__wrap0__seed{seed}__v{date_tag}__{time_tag}"

    run_tag = None
    if seeds_key:
        # keep ONLY the part after ":", so "12x12_K6:runA" -> "runA"
        run_tag = seeds_key.split(":", 1)[1] if ":" in seeds_key else seeds_key
        safe_run = "".join(c if c.isalnum() or c in "-_+" else "-" for c in run_tag)
        base = f"{base}__seeds-{safe_run}"
    
    # Use resolved outdir
    outdir = Path(outdir_val); outdir.mkdir(parents=True, exist_ok=True)

    nodes = build_nodes(H, W)
    edges = build_edges_grid(H, W)

    nodes_p = outdir / f"{base}__nodes.csv"
    edges_p = outdir / f"{base}__edges_grid.csv"
    nodes.to_csv(nodes_p, index=False)
    edges.to_csv(edges_p, index=False)

    # Print using resolved H/W
    print(f"[OK] nodes: {nodes.shape} → {nodes_p.name}")
    print(f"[OK] edges_grid: {edges.shape} (expected {H*(W-1)+W*(H-1)}) → {edges_p.name}")

    if args.write_seeds:
        preset = None
        if seeds_file:
            preset = load_seeds_from_file(seeds_file, H, W, K, key=seeds_key, rng=seed)
        seeds_xy = choose_seeds(H, W, K, preset=preset, rng=seed)
        suffix = f"partition_{(run_tag or 'init')}"
        part_p = write_partition_init(
            nodes, H, W, K, seeds_xy, base, outdir,
            plan_name=(run_tag or "init"), fname_suffix=suffix
        )
        nodes.to_csv(nodes_p, index=False)
        print(f"[OK] partition_init: {part_p.name} (K={K} seeds set)")

    print("\nNext: (optionally) opinions (HBO/iid/blobs), then BFS/Djikstra's validation.")

if __name__ == "__main__":
    main()
