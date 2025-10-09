# Redo experiment per request:
# - Target opinion is fixed at 7 for everyone (metric only)
# - Leaders are NOT fixed to 7; they start like others and update like others.
# - Representative edges weight = 1.
# - Compare Z = Σ|w| vs Z = Σ|w|·|μ| for the three DRFs.

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd


rng = np.random.default_rng(23)

# ----- Graph & setup -----
N = 150
m_edges = 3
G = nx.barabasi_albert_graph(N, m_edges, seed=12).to_directed()

K = 6
leader_ids = rng.choice(N, size=K, replace=False)

# 2) Gaussian around 3.5 (σ=1.0), clipped to [0,7]
opin0 = np.clip(rng.normal(3.5, 1.0, size=N), 0, 7)

# Assign each node to a leader for rep edge
leader_map = rng.choice(leader_ids, size=N)
leader_map[leader_ids] = leader_ids

# weights: rep=1 as requested
w_peer = 1.0
w_rep  = 1.0

neighbors = [[] for _ in range(N)]
weights   = [[] for _ in range(N)]
for u, v in G.edges():
    neighbors[v].append(u)
    weights[v].append(w_peer)
for v in range(N):
    rep = leader_map[v]
    if rep != v:
        neighbors[v].append(rep)
        weights[v].append(w_rep)

targets = np.full(N, 7.0)

# ----- DRFs (user's definitions) -----
def drf_f1(discrepancy):
    delta = abs(discrepancy)
    if 0 <= delta <= 1:
        return 0
    elif 1 < delta <= 2:
        return delta - 1
    elif 2 < delta <= 3:
        return 1
    elif 3 < delta <= 3.2:
        return -2 * delta + 7
    elif 3.2 < delta < 4:
        return 0
    elif 4 <= delta < 5:
        return -1
    elif 5 <= delta < 6:
        return delta - 6
    elif 6 <= delta:
        return 0

def drf_inchworm_withso(discrepancy):
    delta = abs(discrepancy)
    if 0 <= delta < 2:
        return 0
    elif 2 <= delta < 4:
        return 1
    elif 4 <= delta < 6:
        return -1
    elif 6 <= delta:
        return 0
    elif delta <= 2:
        return 0

def drf_inc_noso(discrepancy):
    delta = abs(discrepancy)
    if 0 == delta:
        return 0
    elif 0 < delta < 3:
        return 1
    elif 3 <= delta < 10:
        return -1
    elif 10 <= delta:
        return 0
    elif delta == 0:
        return 0

def step_update(opin, drf, z_mode="all", eta=1.0):
    new = opin.copy()
    for v in range(N):
        num = 0.0
        Z = 0.0
        x_v = opin[v]
        for u, w in zip(neighbors[v], weights[v]):
            x_u = opin[u]  # leaders are not fixed; just use current opinions
            mu = drf(abs(x_u - x_v))
            direction = np.sign(x_u - x_v)
            num += w * mu * direction
            if z_mode == "all":
                Z += abs(w)
            elif z_mode == "eff":
                Z += abs(w) * abs(mu)
            else:
                raise ValueError("z_mode must be 'all' or 'eff'")
        if Z > 0:
            new[v] = x_v + eta * (num / Z)
    return new

def run_experiment(drf, drf_name, T=100, eta=1.0):
    results = {}
    for z_mode in ["all", "eff"]:
        opin = opin0.copy()
        traj = [opin.copy()]
        mean_dev = []
        for t in range(T):
            opin = step_update(opin, drf, z_mode=z_mode, eta=eta)
            traj.append(opin.copy())
            mean_dev.append(np.mean(np.abs(opin - targets)))
        results[z_mode] = {"traj": traj, "mean_dev": mean_dev}
    # Plot mean deviation curves
    plt.figure(figsize=(7,4))
    plt.plot(range(1, T+1), results["all"]["mean_dev"], label="Z = Σ|w|")
    plt.plot(range(1, T+1), results["eff"]["mean_dev"], label="Z = Σ|w|·|μ|")
    plt.xlabel("step")
    plt.ylabel("mean |opinion - 7|")
    plt.title(f"{drf_name}: distance-to-7 with two Z choices (leaders not fixed)")
    plt.legend()
    plt.tight_layout()
    plt.show()
    # Plot total movement hist
    delta_all = np.array(results["all"]["traj"][-1]) - np.array(results["all"]["traj"][0])
    delta_eff = np.array(results["eff"]["traj"][-1]) - np.array(results["eff"]["traj"][0])
    plt.figure(figsize=(7,4))
    bins = np.linspace(-3, 3, 40)
    plt.hist(delta_all, bins=bins, alpha=0.6, label="Z = Σ|w|")
    plt.hist(delta_eff, bins=bins, alpha=0.6, label="Z = Σ|w|·|μ|")
    plt.xlabel("Δ opinion over 100 steps")
    plt.ylabel("count")
    plt.title(f"{drf_name}: total movement after 100 steps (leaders not fixed)")
    plt.legend()
    plt.tight_layout()
    plt.show()
    # Stats table
    stats = pd.DataFrame({
        "metric": ["mean |x-7| @T", "median |x-7| @T", "mean |Δ|", "p(|Δ|<0.1)"],
        "Z = Σ|w|": [
            float(np.mean(np.abs(results["all"]["traj"][-1] - targets))),
            float(np.median(np.abs(results["all"]["traj"][-1] - targets))),
            float(np.mean(np.abs(delta_all))),
            float(np.mean(np.abs(delta_all) < 0.1)),
        ],
        "Z = Σ|w|·|μ|": [
            float(np.mean(np.abs(results["eff"]["traj"][-1] - targets))),
            float(np.median(np.abs(results["eff"]["traj"][-1] - targets))),
            float(np.mean(np.abs(delta_eff))),
            float(np.mean(np.abs(delta_eff) < 0.1)),
        ],
    })
    return stats

stats_f1 = run_experiment(drf_f1, "drf_f1 (piecewise)")
stats_withso = run_experiment(drf_inchworm_withso, "drf_inchworm_withso")
stats_noso = run_experiment(drf_inc_noso, "drf_inc_noso")
