# grid_graphs.py — local visualization for geometry-style grids (x→right, y→up)
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def resolve_partition_path(in_dir: Path, base: str, suffix: str | None):
    """Return the partition CSV path.
    If suffix is None or 'auto', search for files like '{base}__partition_*.csv'.
    Pick the newest if multiple are present. Fall back to 'partition_init.csv'."""
    if suffix and suffix.lower() != "auto":
        p = in_dir / f"{base}__{suffix}.csv"
        if not p.exists():
            raise FileNotFoundError(f"Partition file not found: {p}")
        return p

    # Auto mode
    candidates = list(in_dir.glob(f"{base}__partition_*.csv"))
    if not candidates:
        # legacy fallback
        p = in_dir / f"{base}__partition_init.csv"
        if p.exists():
            return p
        raise FileNotFoundError(
            f"No partition files found for base '{base}' in '{in_dir}'. "
            "Looked for '__partition_*.csv' and '__partition_init.csv'."
        )
    if len(candidates) > 1:
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]

def load_data(base, in_dir, partition_suffix=None):
    in_dir = Path(in_dir)
    nodes = pd.read_csv(in_dir / f"{base}__nodes.csv")
    edges = pd.read_csv(in_dir / f"{base}__edges_grid.csv")
    part_path = resolve_partition_path(in_dir, base, partition_suffix)
    part  = pd.read_csv(part_path)

    df = nodes.merge(part[["id","district","plan_name"]], on="id", how="left")
    if "district" not in df or df["district"].isna().any():
        df["district"] = df["district"].fillna(-1).astype(int)

    print(f"[viz] using partition: {part_path.name}")
    return df, edges


def infer_dims(df_nodes):
    W = int(df_nodes["x"].max()) + 1
    H = int(df_nodes["y"].max()) + 1
    return W, H

def build_arrays(df):
    W, H = infer_dims(df)
    assign = np.full((H, W), -1, dtype=int)
    opin   = np.zeros((H, W), dtype=float)
    for _, r in df.iterrows():
        x, y = int(r["x"]), int(r["y"])
        assign[y, x] = int(r["district"])
        opin[y, x]   = float(r["opinion"])
    return assign, opin

def draw_district_map(assign, df, out_path):
    H, W = assign.shape
    # Mask -1 so unassigned appears light
    masked = np.ma.masked_where(assign < 0, assign)
    plt.figure(figsize=(6,6))
    im = plt.imshow(masked, origin="lower", interpolation="nearest")
    plt.title("District Map (seeded plan)" if (assign>=0).sum() else "District Map (unlabeled)")
    plt.xlabel("x"); plt.ylabel("y")
    # gridlines at cell boundaries
    ax = plt.gca()
    ax.set_xticks(np.arange(-0.5, W, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, H, 1), minor=True)
    ax.grid(which="minor", linestyle=":", linewidth=0.5)

    # Draw district boundaries (thicker black lines where neighbor labels differ)
    # Vertical boundaries between x and x+1 at x+0.5
    for y in range(H):
        for x in range(W-1):
            if assign[y, x] != assign[y, x+1]:
                plt.vlines(x+0.5, y-0.5, y+0.5, linewidth=1.5)

    # Horizontal boundaries between y and y+1 at y+0.5
    for y in range(H-1):
        for x in range(W):
            if assign[y, x] != assign[y+1, x]:
                plt.hlines(y+0.5, x-0.5, x+0.5, linewidth=1.5)

    # Annotate seeds (district >= 0) at their centers
    seeds = df[df["district"] >= 0][["x","y","district"]]
    for _, r in seeds.iterrows():
        plt.text(r["x"], r["y"], str(int(r["district"])),
                 ha="center", va="center", fontsize=9, weight="bold")

    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()

def draw_grid_graph(df_nodes, edges, out_path):
    plt.figure(figsize=(6,6))
    xy = df_nodes.set_index("id")[["x","y"]]
    # edges
    for _, e in edges.iterrows():
        u, v = int(e["u"]), int(e["v"])
        x0, y0 = xy.loc[u, "x"], xy.loc[u, "y"]
        x1, y1 = xy.loc[v, "x"], xy.loc[v, "y"]
        plt.plot([x0, x1], [y0, y1], linewidth=0.5)
    # nodes
    plt.scatter(df_nodes["x"], df_nodes["y"], s=10)
    # seeds larger with labels
    seeds = df_nodes[df_nodes["seed_flag"] == 1]
    if not seeds.empty:
        plt.scatter(seeds["x"], seeds["y"], s=40)
        # if you want to label with district numbers, merge partition first
    plt.title("Grid Graph (4-neighbor) with Seeds")
    plt.xlabel("x"); plt.ylabel("y")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()

def draw_opinion_heatmap(opin, out_path):
    H, W = opin.shape
    plt.figure(figsize=(6,6))
    plt.imshow(opin, origin="lower", interpolation="nearest")
    plt.title("Opinion Field")
    plt.xlabel("x"); plt.ylabel("y")
    ax = plt.gca()
    ax.set_xticks(np.arange(-0.5, W, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, H, 1), minor=True)
    ax.grid(which="minor", linestyle=":", linewidth=0.5)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition-suffix", default="auto",
    help="Partition suffix (default: auto). Examples: 'partition_runA', 'partition_init'")
    ap.add_argument("--base", required=True,
                    help="BASE prefix, e.g., grid_12x12__K6__adj4__wrap0__seed42__v20250902")
    ap.add_argument("--in-dir", default=".",
                    help="Folder containing the CSVs (default: current dir)")
    ap.add_argument("--out-dir", default="viz",
                    help="Where to save PNGs (default: ./viz)")
    args = ap.parse_args()

    df, edges = load_data(args.base, args.in_dir, args.partition_suffix)
    assign, opin = build_arrays(df)

    out_dir = Path(args.out_dir)
    draw_district_map(assign, df, out_dir / f"{args.base}__district_map.png")
    draw_grid_graph(df[["id","x","y","seed_flag"]], edges, out_dir / f"{args.base}__grid_graph.png")
    draw_opinion_heatmap(opin, out_dir / f"{args.base}__opinion.png")

    # quick console summary
    K_seeds = (df["district"] >= 0).sum()
    print(f"[viz] saved to: {out_dir}")
    print(f"[summary] nodes={len(df)}, edges={len(edges)}, seeds_labeled={K_seeds}")

if __name__ == "__main__":
    main()
