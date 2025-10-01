# grid_graphs.py — local visualization for geometry-style grids (x→right, y→up)
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm, LightSource
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


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
    """
    Color cells by opinion threshold: >0.50 = soft red, <=0.50 = soft blue,
    and write the opinion value (2 decimals) in each cell.
    """
    H, W = opin.shape
    # 0 = <=0.5 (blue), 1 = >0.5 (red)
    mask = (opin > 0.5).astype(int)

    # Soft colors
    cmap = ListedColormap(["#9ecae1", "#fcbba1"])  # light blue, light salmon
    norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)

    plt.figure(figsize=(6, 6))
    plt.imshow(mask, origin="lower", interpolation="nearest", cmap=cmap, norm=norm)
    plt.title("Opinion Field (blue ≤ 0.50, red > 0.50)")
    plt.xlabel("x"); plt.ylabel("y")

    # gridlines at cell boundaries
    ax = plt.gca()
    ax.set_xticks(np.arange(-0.5, W, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, H, 1), minor=True)
    ax.grid(which="minor", linestyle=":", linewidth=0.5)

    # annotate opinions with 2 decimals
    for y in range(H):
        for x in range(W):
            plt.text(x, y, f"{opin[y, x]:.2f}",
                     ha="center", va="center", fontsize=7, color="#222222")

    # simple legend
    handles = [Patch(color="#9ecae1", label="≤ 0.50"),
               Patch(color="#fcbba1", label="> 0.50")]
    # plt.legend(handles=handles, loc="upper right", frameon=False)

    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()

def draw_opinion_dem(opin, out_path_png, azdeg=315, altdeg=45,
                     add_contours=True, annotate=False, contour_step=0.1):
    """
    DEM-style shaded relief for the opinion field:
      - elevation Z = opinion in [0,1]
      - base tint: blue (≤0.5) / red (>0.5)
      - hillshade lighting from (azdeg, altdeg)
    """
    H, W = opin.shape
    # Build RGB base by threshold
    red  = np.array([252/255, 187/255, 161/255])  # #fcbba1 (soft red)
    blue = np.array([158/255, 202/255, 225/255])  # #9ecae1 (soft blue)
    base_rgb = np.where((opin > 0.5)[..., None], red, blue)  # (H,W,3)

    # Hillshade using opinions as elevation
    ls = LightSource(azdeg=azdeg, altdeg=altdeg)
    # 'shade_rgb' drapes lighting over the RGB base using the elevation surface
    shaded = ls.shade_rgb(base_rgb, opin, vert_exag=1.0, fraction=0.8)

    plt.figure(figsize=(7, 6))
    ax = plt.gca()
    ax.imshow(shaded, origin="lower", interpolation="nearest")
    ax.set_title("Opinion DEM (hillshade: az={}°, alt={}°)".format(azdeg, altdeg))
    ax.set_xlabel("x"); ax.set_ylabel("y")

    # Cell gridlines
    ax.set_xticks(np.arange(-0.5, W, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, H, 1), minor=True)
    ax.grid(which="minor", linestyle=":", linewidth=0.5, alpha=0.5)

    # Optional contours on top (hypsometric lines)
    if add_contours:
        levels = np.arange(0, 1 + 1e-9, contour_step)
        X, Y = np.meshgrid(np.arange(W), np.arange(H))
        cs = ax.contour(X, Y, opin, levels=levels, colors="k", linewidths=0.4, alpha=0.5)
        ax.clabel(cs, fmt="%.1f", fontsize=7, inline=True)  # label every few

    # Optional per-cell numeric labels
    if annotate:
        for y in range(H):
            for x in range(W):
                ax.text(x, y, f"{opin[y, x]:.2f}",
                        ha="center", va="center", fontsize=7, color="#222222")

    # Legend patches
    handles = [Patch(color="#9ecae1", label="≤ 0.50"),
               Patch(color="#fcbba1", label="> 0.50")]
    ax.legend(handles=handles, loc="upper right", frameon=False)

    plt.tight_layout()
    Path(out_path_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path_png, dpi=180)
    plt.close()

def draw_opinion_surface3d_html(opin, out_path_html):
    """
    Interactive (mouse-rotatable) 3D using Plotly. Requires `pip install plotly`.
    """
    try:
        import plotly.graph_objects as go
    except Exception as e:
        print("[warn] plotly not installed; skipping interactive 3D:", e)
        return

    H, W = opin.shape
    x = np.arange(W); y = np.arange(H)

    # two-tone colorscale with a hard threshold at 0.5
    colorscale = [
        [0.00, "#9ecae1"],  # blue
        [0.50, "#9ecae1"],
        [0.500001, "#fcbba1"],  # red
        [1.00, "#fcbba1"],
    ]

    fig = go.Figure(data=[
        go.Surface(x=x, y=y, z=opin, colorscale=colorscale, cmin=0, cmax=1, showscale=False)
    ])
    fig.update_layout(
        scene=dict(
            xaxis_title="x", yaxis_title="y", zaxis_title="opinion",
            zaxis=dict(range=[0,1])
        ),
        template="plotly_white",
        width=900, height=750,
        margin=dict(l=10, r=10, t=30, b=10)
    )
    Path(out_path_html).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path_html), include_plotlyjs="cdn")
    print(f"[viz] interactive 3D → {out_path_html}")


def make_opinion_spin_gif(opin, out_path_gif, frames=180, elev=35, fps=15):
    try:
        import imageio.v2 as imageio
    except Exception as e:
        print("[warn] imageio not installed; skipping spin GIF:", e)
        return

    H, W = opin.shape
    X, Y = np.meshgrid(np.arange(W), np.arange(H))
    red  = np.array([252/255, 187/255, 161/255, 1.0])
    blue = np.array([158/255, 202/255, 225/255, 1.0])
    facecolors = np.where(opin[..., None] > 0.5, red, blue)

    images = []
    for azim in np.linspace(0, 360, frames, endpoint=False):
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, opin, rstride=1, cstride=1,
                        facecolors=facecolors, linewidth=0, antialiased=True, shade=False)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("opinion")
        ax.set_zlim(0, 1); ax.view_init(elev=elev, azim=azim)

        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        if hasattr(fig.canvas, "buffer_rgba"):
            buf = fig.canvas.buffer_rgba()
            img = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)[..., :3].copy()
        else:
            argb = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8).reshape(h, w, 4)
            rgba = np.roll(argb, -1, axis=2)
            img = rgba[..., :3].copy()

        images.append(img)
        plt.close(fig)

    Path(out_path_gif).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out_path_gif, images, fps=fps)
    print(f"[viz] spin GIF → {out_path_gif}")
    
def draw_opinion_pyramids3d(opin, out_path_png, base_size=0.9, elev=35, azim=45,
                            edge_alpha=0.15, face_alpha=1.0):
    """
    Draw one square-based pyramid per cell:
      - base centered at (x,y), side length = base_size (≤1.0)
      - apex at (x,y, Z) with Z = opinion in [0,1]
      - color: soft blue if ≤0.50, soft red if >0.50
    """
    H, W = opin.shape
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")

    # colors
    RED  = (252/255, 187/255, 161/255, face_alpha)  # #fcbba1
    BLUE = (158/255, 202/255, 225/255, face_alpha)  # #9ecae1
    EDGE = (0, 0, 0, edge_alpha)

    s = float(base_size) / 2.0  # half side
    faces = []
    fcols = []

    # Build triangles for each cell
    for y in range(H):
        for x in range(W):
            z = float(opin[y, x])
            c = RED if z > 0.5 else BLUE

            # base corners on z=0 (centered at (x,y))
            v0 = (x - s, y - s, 0.0)
            v1 = (x + s, y - s, 0.0)
            v2 = (x + s, y + s, 0.0)
            v3 = (x - s, y + s, 0.0)
            apex = (x, y, z)

            # 4 side triangles (we skip the base to keep it visually clean)
            faces.extend([
                [v0, v1, apex],
                [v1, v2, apex],
                [v2, v3, apex],
                [v3, v0, apex],
            ])
            fcols.extend([c, c, c, c])

    coll = Poly3DCollection(faces, facecolors=fcols, edgecolor=EDGE, linewidths=0.4)
    ax.add_collection3d(coll)

    # Axes and view
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(-0.5, H - 0.5)
    ax.set_zlim(0.0, 1.0)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("opinion")
    ax.view_init(elev=elev, azim=azim)

    # Optional thin grid on the floor
    ax.plot([ -0.5,  W-0.5], [-0.5, -0.5], [0,0], color=(0,0,0,0.2))
    ax.plot([ -0.5,  W-0.5], [ H-0.5, H-0.5], [0,0], color=(0,0,0,0.2))
    ax.plot([ -0.5, -0.5], [ -0.5,  H-0.5], [0,0], color=(0,0,0,0.2))
    ax.plot([ W-0.5, W-0.5], [ -0.5,  H-0.5], [0,0], color=(0,0,0,0.2))

    plt.tight_layout()
    Path(out_path_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path_png, dpi=180)
    plt.close(fig)

def draw_opinion_pyramids3d_html(opin, out_path_html, base_size=0.9):
    """
    Interactive HTML with Plotly Mesh3d pyramids (rotate with mouse).
    """
    try:
        import plotly.graph_objects as go
    except Exception as e:
        print("[warn] plotly not installed; skipping interactive pyramids:", e)
        return

    H, W = opin.shape
    s = float(base_size) / 2.0

    xs, ys, zs = [], [], []
    I, J, K = [], [], []  # triangle indices
    intens = []           # use opinion to drive a 0.5-threshold colorscale

    def add_tri(a, b, c):
        I.append(a); J.append(b); K.append(c)

    vidx = 0
    for y in range(H):
        for x in range(W):
            z = float(opin[y, x])

            # 5 vertices per pyramid: v0..v3 base (z=0), v4 apex (z=z)
            v0 = (x - s, y - s, 0.0)
            v1 = (x + s, y - s, 0.0)
            v2 = (x + s, y + s, 0.0)
            v3 = (x - s, y + s, 0.0)
            v4 = (x, y, z)

            verts = [v0, v1, v2, v3, v4]
            for vx, vy, vz in verts:
                xs.append(vx); ys.append(vy); zs.append(vz); intens.append(z)

            # 4 side faces (triangles). We'll skip base faces for clarity.
            add_tri(vidx+0, vidx+1, vidx+4)
            add_tri(vidx+1, vidx+2, vidx+4)
            add_tri(vidx+2, vidx+3, vidx+4)
            add_tri(vidx+3, vidx+0, vidx+4)

            vidx += 5

    # Two-tone colorscale with a hard step at 0.5
    colorscale = [
        [0.00, "#9ecae1"],   # blue
        [0.50, "#9ecae1"],
        [0.500001, "#fcbba1"],  # red
        [1.00, "#fcbba1"]
    ]

    mesh = go.Mesh3d(
        x=xs, y=ys, z=zs,
        i=I,  j=J,  k=K,
        intensity=intens, colorscale=colorscale, cmin=0, cmax=1,
        showscale=False, flatshading=True, opacity=1.0
    )

    fig = go.Figure(data=[mesh])
    fig.update_layout(
        scene=dict(
            xaxis_title="x", yaxis_title="y", zaxis_title="opinion",
            xaxis=dict(range=[-0.5, W-0.5]),
            yaxis=dict(range=[-0.5, H-0.5]),
            zaxis=dict(range=[0, 1]),
            # ▼ stretch Z visually (no data change)
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=0.75)  # try 2.0 or 3.0 for more height
        ),
        template="plotly_white",
        width=900, height=750, margin=dict(l=10,r=10,t=30,b=10)
    )

    Path(out_path_html).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path_html), include_plotlyjs="cdn")
    print(f"[viz] interactive pyramids → {out_path_html}")

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
    ap.add_argument("--opinion-style",
                choices=["binary", "dem"],
                default="binary",
                help="2D opinion render: 'binary' (blue/red) or 'dem' (hillshade)")

    ap.add_argument("--interactive-3d", action="store_true",
                    help="Write rotatable 3D HTML (requires plotly)")

    ap.add_argument("--spin", type=int, default=0,
                    help="If >0, make a spin GIF with this many frames (requires imageio)")

    # DEM knobs (used only when --opinion-style dem)
    ap.add_argument("--azdeg", type=float, default=315.0,
                    help="Hillshade azimuth (degrees)")
    ap.add_argument("--altdeg", type=float, default=45.0,
                    help="Hillshade altitude (degrees)")
    ap.add_argument("--dem-contours", action="store_true",
                    help="Overlay contours on DEM")
    ap.add_argument("--dem-annotate", action="store_true",
                    help="Annotate DEM cells with 2-dec values")
    
    # Pyramid knobs
    ap.add_argument("--peaks3d", action="store_true",
                help="Render pyramid peaks (static PNG)")
    ap.add_argument("--peaks3d-html", action="store_true",
                    help="Render rotatable pyramid peaks (HTML; requires plotly)")
    ap.add_argument("--peak-base", type=float, default=0.9,
                    help="Pyramid base size in cell units (0<base≤1)")
    ap.add_argument("--peak-elev", type=float, default=35.0,
                    help="Matplotlib 3D elevation angle")
    ap.add_argument("--peak-azim", type=float, default=45.0,
                    help="Matplotlib 3D azimuth angle")


    args = ap.parse_args()

    df, edges = load_data(args.base, args.in_dir, args.partition_suffix)
    assign, opin = build_arrays(df)

    out_dir = Path(args.out_dir)
    draw_district_map(assign, df, out_dir / f"{args.base}__district_map.png")
    draw_grid_graph(df[["id","x","y","seed_flag"]], edges, out_dir / f"{args.base}__grid_graph.png")
    # ---- OPINION RENDERING ----
    out_dir = Path(args.out_dir)
    stem = args.base

    # 2D map (pick style)
    if args.opinion_style == "dem":
        draw_opinion_dem(
            opin,
            out_dir / f"{stem}__opinion_dem.png",
            azdeg=args.azdeg, altdeg=args.altdeg,
            add_contours=args.dem_contours,
            annotate=args.dem_annotate
        )
    elif args.opinion_style == "binary":
        draw_opinion_heatmap(opin, out_dir / f"{stem}__opinion.png")
    else:
        draw_opinion_heatmap(opin, out_dir / f"{stem}__opinion.png")

    # Optional 3D SURFACE (continuous surface; rotatable HTML & spin GIF)
    if args.interactive_3d:
        draw_opinion_surface3d_html(opin, out_dir / f"{stem}__opinion3d.html")
    if args.spin and args.spin > 0:
        make_opinion_spin_gif(opin, out_dir / f"{stem}__opinion3d_spin.gif",
                            frames=args.spin)

    # Optional 3D PYRAMID PEAKS (your “mountain” per cell)
    if args.peaks3d:
        draw_opinion_pyramids3d(
            opin,
            out_dir / f"{stem}__opinion_peaks3d.png",
            base_size=args.peak_base,
            elev=args.peak_elev,
            azim=args.peak_azim
        )
    if args.peaks3d_html:
        draw_opinion_pyramids3d_html(
            opin,
            out_dir / f"{stem}__opinion_peaks3d.html",
            base_size=args.peak_base
        )

    # quick console summary
    K_seeds = (df["district"] >= 0).sum()
    print(f"[viz] saved to: {out_dir}")
    print(f"[summary] nodes={len(df)}, edges={len(edges)}, seeds_labeled={K_seeds}")

if __name__ == "__main__":
    main()
