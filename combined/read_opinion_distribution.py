import re, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import zarr

# If DRF_f1 then use these paths
# EXP_PATH = r"F:\Carleton University\Prof Zinovi RA\Code\artifacts_zarr_drf_1\exp-K6_H8xW9_F30_M30_S100_drf_f1_mad.zarr"
# RUN_NAME = "run-20251008-093608-b03dc421"   # or "run-20251008-..." if you used time-based ids

# If DRF_f4 then use these paths
EXP_PATH = r"F:\Carleton University\Prof Zinovi RA\Code\artifacts_zarr_drf_4\exp-K6_H8xW9_F30_M30_S100_drf_f4_mad.zarr"
RUN_NAME = "run-20251008-045422-99883f15"   # or "run-20251008-..." if you used time-based ids

root = zarr.open_group(EXP_PATH, mode="r")
g_run = root["runs"][RUN_NAME]

def infer_drf_name(root, exp_path: str) -> str:
    # 1) Try Zarr attrs (best)
    drf = root.attrs.get("drf_name")
    if isinstance(drf, (str, bytes)) and len(str(drf).strip()) > 0:
        return str(drf)

    # 2) Fallback: parse from EXP_PATH (e.g., "..._drf_f4_mad.zarr")
    m = re.search(r"drf_([^/\\]+)", exp_path)
    if m:
        return m.group(1)

    return "unknown_drf"

DRF_NAME = infer_drf_name(root, EXP_PATH)

rows = []
for f_key in sorted([k for k in g_run.keys() if k.startswith("f_")]):
    g_f = g_run[f_key]
    for m_key in sorted([k for k in g_f.keys() if k.startswith("m_")]):
        g_m = g_f[m_key]
        for c_key in sorted([k for k in g_m.keys() if k.startswith("c_")]):
            g_c = g_m[c_key]
            init_mad  = float(g_c["world_init"].attrs["init_dist_mean_mad"])
            final_mad = float(g_c["summary"].attrs["final_mean_mad_to_cstar"])
            delta     = float(g_c["summary"].attrs["delta_mean_mad"])
            rows.append({"run": RUN_NAME, "f": f_key, "m": m_key, "c": c_key,
                         "init_mad": init_mad, "final_mad": final_mad, "delta_mad": delta})

df = pd.DataFrame(rows).sort_values(["f","m","c"])
print(df.shape)    # expect (F*M_per_f*len(c_star_list), 7) e.g., (2700, 7)

# you can save to CSV if you want
# If using drf_f1 then use this path
# df.to_csv(f"F:/Carleton University/Prof Zinovi RA/Code/artifacts_zarr_drf_1/exp-K6_H8xW9_F30_M30_S100_drf_f1_mad.zarr/runs/run-20251008-093608-b03dc421/csv_and_histogram/{RUN_NAME}.csv", index=False)

# If using drf_f4 then use this path
df.to_csv(f"F:/Carleton University/Prof Zinovi RA/Code/artifacts_zarr_drf_4/exp-K6_H8xW9_F30_M30_S100_drf_f4_mad.zarr/runs/run-20251008-045422-99883f15/csv_and_histogram/{RUN_NAME}.csv", index=False)

# 1) add a numeric c index and (optionally) the actual c* value
df = df.copy()
df["c_idx"] = df["c"].str.extract(r"c_(\d+)").astype(int)

# optional: pull c* list from experiment root attrs, if present
try:
    cstar_list = list(map(float, root.attrs.get("c_star_list", [])))
    df["c_star"] = df["c_idx"].map(lambda i: cstar_list[i] if i < len(cstar_list) else np.nan)
    c_label_fmt = lambda r: f"c_idx={r['c_idx']} (c*={r['c_star']})"
except Exception:
    c_label_fmt = lambda r: f"c_idx={r['c_idx']}"

# 2) plotting helper
def plot_metric_by_c(df, metric, outdir=".", bins=30, drf=DRF_NAME):
    cs = sorted(df["c_idx"].unique())
    # common range across c for fair comparison
    vmin = df[metric].min()
    vmax = df[metric].max()
    # small padding
    pad = 0.02 * (vmax - vmin if vmax > vmin else 1.0)
    rng = (vmin - pad, vmax + pad)

    fig, axes = plt.subplots(1, len(cs), figsize=(5*len(cs), 4), squeeze=False)
    axes = axes[0]

    for ax, c in zip(axes, cs):
        dd = df[df["c_idx"] == c][metric].to_numpy()
        ax.hist(dd, bins=bins, range=rng)
        # title
        if "c_star" in df.columns:
            cstar_val = df[df["c_idx"]==c]["c_star"].iloc[0]
            ax.set_title(f"c={c} (c*={cstar_val:g})")
        else:
            ax.set_title(f"c={c}")
        ax.set_xlabel(metric)
        ax.set_ylabel("count")
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Histogram of {metric} by c for DRF: {drf}")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    os.makedirs(outdir, exist_ok=True)
    fname = os.path.join(outdir, f"hist_{metric}_by_c.png")
    fig.savefig(fname, dpi=150)
    print(f"saved {fname}")

# 3) make the three figures
# If using drf_f1 then use this path
# plot_metric_by_c(df, "init_mad",  outdir=f"F:/Carleton University/Prof Zinovi RA/Code/artifacts_zarr_drf_1/exp-K6_H8xW9_F30_M30_S100_drf_f1_mad.zarr/runs/run-20251008-093608-b03dc421/csv_and_histogram")
# plot_metric_by_c(df, "final_mad", outdir=f"F:/Carleton University/Prof Zinovi RA/Code/artifacts_zarr_drf_1/exp-K6_H8xW9_F30_M30_S100_drf_f1_mad.zarr/runs/run-20251008-093608-b03dc421/csv_and_histogram")
# plot_metric_by_c(df, "delta_mad", outdir=f"F:/Carleton University/Prof Zinovi RA/Code/artifacts_zarr_drf_1/exp-K6_H8xW9_F30_M30_S100_drf_f1_mad.zarr/runs/run-20251008-093608-b03dc421/csv_and_histogram")

# If using drf_f4 then use this path
plot_metric_by_c(df, "init_mad",  outdir=f"F:/Carleton University/Prof Zinovi RA/Code/artifacts_zarr_drf_4/exp-K6_H8xW9_F30_M30_S100_drf_f4_mad.zarr/runs/run-20251008-045422-99883f15/csv_and_histogram")
plot_metric_by_c(df, "final_mad", outdir=f"F:/Carleton University/Prof Zinovi RA/Code/artifacts_zarr_drf_4/exp-K6_H8xW9_F30_M30_S100_drf_f4_mad.zarr/runs/run-20251008-045422-99883f15/csv_and_histogram")
plot_metric_by_c(df, "delta_mad", outdir=f"F:/Carleton University/Prof Zinovi RA/Code/artifacts_zarr_drf_4/exp-K6_H8xW9_F30_M30_S100_drf_f4_mad.zarr/runs/run-20251008-045422-99883f15/csv_and_histogram")