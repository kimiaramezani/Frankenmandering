import zarr, pandas as pd

EXP_PATH = r"artifacts_zarr/exp-K6_H8xW9_F5_M3_S100_drfdrf_fig4_mad.zarr"
RUN_NAME = "run-20251008-041723-82d52537"   # or "run-20251008-..." if you used time-based ids

root = zarr.open_group(EXP_PATH, mode="r")
g_run = root["runs"][RUN_NAME]

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
print(df)    # expect (F*M_per_f*len(c_star_list), 7) e.g., (2700, 7)
