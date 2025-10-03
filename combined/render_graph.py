import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import torch
from torch_geometric.data import Data, HeteroData
from graph_initiator import build_init_data

# --- safe tensor -> numpy (CPU) helper ---
def _to_np(x):
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

# Big categorical palette: concat tab20, tab20b, tab20c (60 distinct hues)
_DISTRICT_BASE = (
    list(mpl.colormaps['tab20'](np.linspace(0, 1, 20))) +
    list(mpl.colormaps['tab20b'](np.linspace(0, 1, 20))) +
    list(mpl.colormaps['tab20c'](np.linspace(0, 1, 20)))
)
# Convert RGBA (0..1) to hex strings for consistency
_DISTRICT_BASE = [mpl.colors.to_hex(c) for c in _DISTRICT_BASE]

# Registry that persists across calls in a single run
_DISTRICT_COLOR_REGISTRY: dict[int, str] = {}

def _district_color(label: int) -> str:
    """Return a stable color for a district label, allocating once and persisting."""
    if label not in _DISTRICT_COLOR_REGISTRY:
        idx = label % len(_DISTRICT_BASE)  # wrap if >60
        _DISTRICT_COLOR_REGISTRY[label] = _DISTRICT_BASE[idx]
    return _DISTRICT_COLOR_REGISTRY[label]

def reset_district_colors():
    """Optional: call at the start of a new *experiment run* to keep runs comparable."""
    _DISTRICT_COLOR_REGISTRY.clear()

def render_graph(data: HeteroData, figsize=(14, 12), return_rgb=False):
    """
    Render a 2x2 grid showing:
    - Top-left: Geo edges + node positions
    - Top-right: Social edges
    - Bottom-left: Opinion values
    - Bottom-right: District assignments
    
    Parameters
    ----------
    data : HeteroData
        PyG heterogeneous graph with:
        - data['node'].pos: (N, 2) positions
        - data['node'].x or opinion_scaled: (N, 1) opinions
        - data['node'].district: (N,) district labels
        - data['node', 'geo', 'node'].edge_index: (2, E_geo)
        - data['node', 'social', 'node'].edge_index: (2, E_social)
    figsize : tuple
        Figure size
    return_rgb : bool
        If True, return (fig, rgb_array), else just fig
        
    Returns
    -------
    fig or (fig, rgb_array)
    """
    
    # Extract data (supports HeteroData or FrankenData/Data)
    if isinstance(data, HeteroData):
        if 'node' not in data.node_types or 'pos' not in data['node']:
            raise ValueError("data['node'].pos is required for HeteroData")
        pos = _to_np(data['node'].pos)  # (N, 2)
        N = pos.shape[0]
        x, y = pos[:, 0], pos[:, 1]

        # Opinion (x, or opinion/opinion_scaled if present)
        opinion = None
        if 'x' in data['node'] and data['node'].x is not None:
            opinion = _to_np(data['node'].x).reshape(N, -1)[:, 0]
        elif 'opinion_scaled' in data['node']:
            opinion = _to_np(data['node'].opinion_scaled).reshape(-1)
        elif 'opinion' in data['node']:
            opinion = _to_np(data['node'].opinion).reshape(-1)

        # Districts
        district = None
        if 'district' in data['node'] and data['node'].district is not None:
            district = _to_np(data['node'].district).astype(int, copy=False).reshape(-1)

        # Edges
        geo_edges = None
        if ('node', 'geo', 'node') in data.edge_types:
            geo_ei = _to_np(data['node', 'geo', 'node'].edge_index)
            geo_edges = geo_ei.T  # (E, 2)

        social_edges = None
        if ('node', 'social', 'node') in data.edge_types:
            soc_ei = _to_np(data['node', 'social', 'node'].edge_index)
            social_edges = soc_ei.T  # (E, 2)

    elif isinstance(data, Data):
        # FrankenData path (torch_geometric.data.Data)
        if not hasattr(data, 'pos'):
            raise ValueError("FrankenData/Data must have .pos")

        pos = _to_np(data.pos)  # (N, 2)
        N = pos.shape[0]
        x, y = pos[:, 0], pos[:, 1]

        # Opinion: stored on data.opinion (N,1)
        opinion = None
        if hasattr(data, 'opinion') and data.opinion is not None:
            opinion = _to_np(data.opinion).reshape(N, -1)[:, 0]

        # Districts: zero-based in FrankenData
        district = None
        if hasattr(data, 'dist_label') and data.dist_label is not None:
            district = _to_np(data.dist_label).astype(int, copy=False).reshape(-1)

        # Edges come from FrankenData fields
        geo_edges = None
        if hasattr(data, 'geographical_edge') and data.geographical_edge is not None:
            ei_geo = _to_np(data.geographical_edge)
            if ei_geo is not None and ei_geo.size > 0:
                geo_edges = ei_geo.T  # (E, 2)

        social_edges = None
        if hasattr(data, 'social_edge') and data.social_edge is not None:
            ei_soc = _to_np(data.social_edge)
            if ei_soc is not None and ei_soc.size > 0:
                social_edges = ei_soc.T  # (E, 2)
    else:
        raise TypeError("Unsupported data type. Expected HeteroData or Data (FrankenData).")

    # Create figure
    fig, axes = plt.subplots(
        2, 2, figsize=figsize, constrained_layout=True,
        sharex='col', sharey='col'
    )
    fig.suptitle('Graph State Visualization', fontsize=16, fontweight='bold')
    
    # --- TOP-LEFT: Geo edges + positions ---
    ax = axes[0, 0]
    ax.set_title('Geographic Layer', fontsize=12, fontweight='bold')
    
    if geo_edges is not None and len(geo_edges) > 0:
        for u, v in geo_edges:
            ax.plot([x[u], x[v]], [y[u], y[v]], 
                   'k-', linewidth=1.2, alpha=0.4, zorder=1)
    
    ax.scatter(x, y, s=60, c='steelblue', edgecolors='darkblue', 
              linewidths=1.5, alpha=0.8, zorder=2)
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.3)
    
    # --- TOP-RIGHT: Social edges ---
    ax = axes[0, 1]
    ax.set_title('Social Layer', fontsize=12, fontweight='bold')
    
    if social_edges is not None and len(social_edges) > 0:
        for u, v in social_edges:
            ax.plot([x[u], x[v]], [y[u], y[v]], 
                   color='tab:orange', linewidth=0.8, alpha=0.3, zorder=1)
    
    ax.scatter(x, y, s=60, c='tab:orange', edgecolors='darkorange', 
              linewidths=1.5, alpha=0.8, zorder=2)
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.3)
    
    # --- BOTTOM-LEFT: Opinions ---
    ax = axes[1, 0]
    ax.set_title('Opinion Distribution', fontsize=12, fontweight='bold')
    
    if opinion is not None:
        # Normalize to [0, 1] for color mapping (robust to constants)
        den = float(opinion.max() - opinion.min())
        if den < 1e-12:
            op_norm = np.zeros_like(opinion, dtype=float)
        else:
            op_norm = (opinion - opinion.min()) / den

        # Use normalized colors with consistent vmin/vmax
        scatter = ax.scatter(
            x, y, s=100, c=op_norm, cmap='RdBu_r',
            edgecolors='black', linewidths=0.8,
            alpha=0.85, zorder=2, vmin=0.0, vmax=7.0
        )

        # Add colorbar
        cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Opinion (0→7)', rotation=270, labelpad=15)

        # Overlay geo edges faintly
        if geo_edges is not None:
            for u, v in geo_edges:
                ax.plot([x[u], x[v]], [y[u], y[v]], 
                       'k-', linewidth=0.5, alpha=0.15, zorder=1)
    else:
        ax.text(0.5, 0.5, 'No opinion data', 
               transform=ax.transAxes, ha='center', va='center',
               fontsize=12, color='gray')
    
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.3)
    
    # --- BOTTOM-RIGHT: Districts ---
    ax = axes[1, 1]
    ax.set_title('District Assignment', fontsize=12, fontweight='bold')
    
    if district is not None:
        uniq = np.sort(np.unique(district))
        # Use persistent registry so each label keeps its color across frames
        label_to_color = {int(lab): _district_color(int(lab)) for lab in uniq}
        node_colors = [label_to_color[int(d)] for d in district]

        ax.scatter(
            x, y, s=100, c=node_colors,
            edgecolors='black', linewidths=0.8,
            alpha=0.85, zorder=2
        )
        if geo_edges is not None:
            for u, v in geo_edges:
                ax.plot([x[u], x[v]], [y[u], y[v]],
                        'k-', linewidth=0.5, alpha=0.15, zorder=1)

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=label_to_color[int(lab)], edgecolor='black',
                label=f'District {int(lab)}')
            for lab in uniq
        ]
        ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.02, 0.5),
                framealpha=0.9, fontsize=9, borderaxespad=0.0)

    else:
        ax.text(0.5, 0.5, 'No district data', transform=ax.transAxes, ha='center', va='center',fontsize=12, color='gray')
    
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.3)
    
    if return_rgb:
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        rgb = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        rgb = rgb.reshape((h, w, 3))
        return fig, rgb
    
    return fig


# Example usage:
# from graph_renderer import render_graph
# 
# # After building your graph:
# graph = Graph(*Graph.make_node_ids(100))
# graph.generate_positions(mode="grid", H=10, W=10)
# graph.build_edges_grid(H=10, W=10, neighborhood="rook")
# graph.build_edges_social_ba(m=2, rng_seed=42)
# graph.fill_opinions_hbo_graph(alpha=2.0, beta=2.0, influence=0.8, scale_out=7.0)
# seeds = graph.choose_random_district_seeds_spaced(K=4, min_manhattan=5)
# graph.greedy_fill_districts(seeds)
# data = graph.to_pyg_hetero()
# 
# # Render
# fig = render_graph(data)
# plt.show()
# 
# # Or get RGB array for RL logging
# fig, rgb = render_graph(data, return_rgb=True)
# # Can save rgb as image or log to tensorboard

init_data, G = build_init_data(
         K=10, H=8, W=9,
         ba_m=2, rng_seed=42,
         neighborhood="rook",
         hbo_alpha=2.0, hbo_beta=2.0, hbo_influence=0.8,
         scale_out=7.0,
         min_manhattan=3,
         attach_hetero=False,
         use_scaled_opinion=True,
     )

fig = render_graph(init_data)
plt.show()