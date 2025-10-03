import matplotlib.pyplot as plt
import networkx as nx
from graph_initiator import build_inchworm_init_data

inch_data, G_inch = build_inchworm_init_data()
pos = {i: (i, 0) for i in range(10)}  # lay nodes 0..9 on a line

chain_edges = [(3,4), (4,5), (5,6), (6,7), (7,8), (8,9)]
spokes = [(0,3), (1,3), (2,3)]
long_arcs = [(3,7), (3,8), (3,9), (4,8), (4,9)]

fig, ax = plt.subplots(constrained_layout=True)  # avoids the tight_layout warning

# nodes + labels once
nx.draw_networkx_nodes(G_inch, pos, ax=ax, node_color="white", edgecolors="black")
nx.draw_networkx_labels(G_inch, pos, ax=ax)

def draw_curved(edgelist, rad, **kw):
    nx.draw_networkx_edges(
        G_inch, pos, edgelist=edgelist, ax=ax,
        arrows=True,            # force FancyArrowPatch
        arrowstyle='-',         # no heads, just a line
        connectionstyle=f"arc3,rad={rad}",
        **kw
    )

# straight-ish chain, small curvature 0.0
draw_curved(chain_edges, 0.00, width=2)
# short “spokes”
draw_curved(spokes, 0.15, width=2)
# long arcs
draw_curved(long_arcs, 0.35, width=2)

ax.set_axis_off()
plt.show()

import math
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch

def plot_opinion_timeline(
    states,
    highlights=None,         # list[set|dict] same length as states (optional)
    arrows=None,             # list[dict[node]->(+1|-1)] same length as states (optional)
    opinion_min=None,        # y-axis min (optional, else min over all states)
    opinion_max=None,        # y-axis max (optional, else max over all states)
    panel_height=8,          # height in opinion units (just affects figsize scaling)
    node_radius=0.22,        # circle radius
    axis_x=0.0,              # x position of the vertical axis within each panel
    x_gap=1.4,               # horizontal spacing between columns (panels)
    fan_dx=0.35,             # horizontal spacing between circles within a row (same y)
    title_prefix="t = "      # panel title prefix
):
    """
    Render a strip of small panels, one per timestep, where y = opinion value and
    circles labeled with node IDs sit at their y. Multiple nodes at same y are fanned out.
    """

    T = len(states)
    if T == 0:
        raise ValueError("states must be a non-empty list")

    # Normalize optional per-timestep inputs
    highlights = highlights or [set() for _ in range(T)]
    arrows     = arrows or [{}     for _ in range(T)]
    # Convert any dict-like highlights to a set of nodes
    norm_high = []
    for h in highlights:
        if isinstance(h, dict):
            norm_high.append({n for n, v in h.items() if v})
        else:
            norm_high.append(set(h))
    highlights = norm_high

    # Determine opinion range
    all_y = []
    for s in states:
        all_y.extend(s.values())
    if not all_y:
        raise ValueError("states contain no opinion values")

    if opinion_min is None:
        opinion_min = math.floor(min(all_y))
    if opinion_max is None:
        opinion_max = math.ceil(max(all_y))

    # Figure size: make each panel narrow and tall
    # height in inches proportional to panel_height; tweak as you like
    h_inches = 3.6
    w_inches = max(3.0, 0.9 * T)  # scale width with number of panels
    fig, axes = plt.subplots(
        1, T, figsize=(w_inches, h_inches), squeeze=False, constrained_layout=True
    )
    axes = axes[0] if T > 1 else [axes[0]]

    # Draw each timestep panel
    for t, ax in enumerate(axes):
        state = states[t]
        hi = highlights[t]
        arr = arrows[t]

        # Axis spine: vertical y-axis with ticks
        ax.axvline(axis_x, color="black", linewidth=1.5)
        ax.annotate("t = " + str(t), xy=(axis_x, opinion_max + 0.3),
                    ha="center", va="bottom", fontsize=10)

        # y ticks as integers across the opinion range
        yticks = list(range(int(opinion_min), int(opinion_max) + 1))
        ax.set_yticks(yticks)
        ax.set_yticklabels([str(y) for y in yticks], fontsize=9)

        # No x ticks; fix limits
        # Leave room to the right for fanned circles
        max_per_row = max(1, max(
            (sum(1 for v in state.values() if v == y) for y in set(state.values())),
            default=1
        ))
        x_min = axis_x - 0.6
        x_max = axis_x + fan_dx * (max_per_row + 1.5)

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(opinion_min - 0.5, opinion_max + 0.8)

        # Build rows: y -> list of nodes at that opinion
        rows = {}
        for n, y in state.items():
            rows.setdefault(y, []).append(n)

        # Sort rows by y, and nodes by id for stable placement
        for y in sorted(rows.keys()):
            nodes_here = sorted(rows[y])
            # Fan them left-to-right with consistent spacing
            for k, n in enumerate(nodes_here):
                cx = axis_x + fan_dx * (k + 1)     # shift right of axis
                cy = y
                face = "gold" if n in hi else "white"
                circ = Circle((cx, cy), radius=node_radius,
                              facecolor=face, edgecolor="black", linewidth=1.2)
                ax.add_patch(circ)
                ax.text(cx, cy, str(n), ha="center", va="center", fontsize=9)

                # Optional arrow marker (e.g., +1 = red ►, -1 = green ►)
                if n in arr:
                    direction = arr[n]
                    color = "red" if direction >= 0 else "green"
                    # Small right-pointing arrow to the right of the circle
                    ax.add_patch(FancyArrowPatch(
                        (cx + node_radius*1.1, cy),
                        (cx + node_radius*1.1 + 0.32, cy),
                        arrowstyle="-|>", mutation_scale=10, linewidth=1.0, color=color
                    ))

        # Clean look
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.set_xticks([])
        ax.set_xlabel("")  # no x label

    return fig, axes

# t=0 from your opinions
state0 = {0:0,1:0,2:0,3:1,4:2,5:3,6:4,7:5,8:5,9:5}

# pretend your algorithm produced a few later states:
state1 = {**state0, 2:1, 3:2, 7:6}         # demo changes
state2 = {**state1, 0:1, 9:6}
state3 = {**state2, 4:3, 8:6}

states = [state0, state1, state2, state3]

# highlight some active movers each step (optional)
highlights = [
    {2,9},      # at t=0 highlight nodes 2 and 9
    {2,3,7},    # at t=1 ...
    {0,9},
    {4,8},
]

# add tiny right arrows colored by direction (+1 up/red, -1 down/green)
# (you can pass whatever semantics you want; this is just a visual cue)
arrows = [
    {9:+1, 2:+1},
    {3:+1, 7:+1},
    {0:+1, 9:+1},
    {8:+1},
]

fig, _ = plot_opinion_timeline(states, highlights=highlights, arrows=arrows)
plt.show()
