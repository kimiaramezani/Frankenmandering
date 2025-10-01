import torch
import torch_geometric
import matplotlib.pyplot as plt
from make_grid_3 import Graph
# print("Torch:", torch.__version__)
# print("CUDA available:", torch.cuda.is_available())
# print("PyG OK:", torch_geometric.__version__)

# # 0) Make nodes (ids 0..N-1) and an empty union graph shell
G = Graph(*Graph.make_node_ids(36))             # N=36 for a 6×6 demo

# # 1) Put nodes on a grid (H×W); you can also use 'uniform_box' etc.
G.generate_positions(mode="grid", H=6, W=6)

# # 2) Build GEO edges (rook or queen). Optional: store both (u,v) and (v,u)
G.build_edges_grid(H=6, W=6, neighborhood="rook", weight_grid=1.0, barrier_flag=0)
# # 2b) You can also build queen edges instead of rook
# # G.build_edges_grid(H=6, W=6, neighborhood="queen", weight_grid=1.0, barrier_flag=0)

# # 3) Build SOCIAL edges (Barabási–Albert). Optional: store both (u,v) and (v,u)
G.build_edges_social_ba(m=2, rng_seed=42, weight_social=1.0)

# # 4) (Optional) Compose a union NetworkX graph for traversal/visualization
G.update_union_graph(carry_layer_flags=True)

# # 5) Fill opinions (HBO-style) on the GEO graph
G.fill_opinions_hbo_graph(alpha=2.0, beta=2.0, influence=0.8, rng_seed=None, scale_out=7.0, scaled_colname="opinion_scaled")

# # 6) (Optional) Districting on the GEO graph
seeds = G.choose_random_district_seeds_spaced(K=3, min_manhattan=3, rng_seed=None)
labels = G.greedy_fill_districts(seeds, rng_seed=None)

# # 7) Export to PyTorch Geometric (HeteroData)
data = G.to_pyg_hetero()

# # ---- quick sanity prints ----
print("GEO edges:", G.G_geo.number_of_edges())
print("SOC edges:", G.G_social.number_of_edges())
print("Union edges:", G.G.number_of_edges())
print("PyG node feat shape (opinions):", data["node"].x.shape)
print("PyG num nodes:", data["node"].num_nodes)
print("PyG positions shape:", data["node"].pos.shape)
print("PyG position type:", data["node"].pos.dtype)  # should be torch.float32
print("PyG GEO edge_index:", data["node","geo","node"].edge_index.shape)
print("PyG GEO edge_index:", data["node","geo","node"].edge_index)
print("PyG SOC edge_index:", data["node","social","node"].edge_index.shape)
print("PyG SOC edge_index:", data["node","social","node"].edge_index)
if ('node','district','node') in data.edge_types:
    print("PyG DIST edge_index:", data[('node','district','node')].edge_index.shape)
    
print("district dtype:", data['node'].district.dtype)           # should be torch.int64
print("district shape:", data['node'].district.shape)           # torch.Size([N])
print("unique districts:", data['node'].district.unique())      # e.g., tensor([1,2,3])
print("counts per district:", torch.bincount(data['node'].district))
print("Seed nodes:", seeds)
print("District labels:", labels)
print("District Label shape:", labels.shape)  # each torch.Size([N])
print("District Label type:", labels.dtype)  # each torch.Size([N])
print("Opinions (x):", data['node'].x.view(-1))  # flatten to 1D
print("Complete data object:", data)

#print("GEO table:\n", G.df_edges_geo)
print("PyG Geo Edges:\n", data['node','geo','node'].edge_index[:,:])
print("SOC table:\n", G.df_edges_social)  # show first 68 rows only
print("PyG Soc Edges:\n", data['node','social','node'].edge_index[:,:])
print("PyG Soc Edges:\n", data['node','social','node'].edge_index[:,:].T) 



# # # ---- 3D multi-layer preview
# # fig = G.render_layers(
# #     show_geo=True,
# #     show_social=True,
# #     show_districts=True,
# #     show_nodes=True,
# #     z_scale_opinion=4.0,   # stretch the opinion surface
# #     z_gap_district=0.60,   # district sheet above
# #     z_gap_social=1.30,     # social edges sheet higher
# #     fill_district_cells=True,
# #     cell_alpha=0.30,
# #     cell_size=0.95,
# #     annotate_district_id=True,
# #     elev=35, azim=-45,
# #     node_size=44,
# #     geo_lw=1.3,
# #     soc_lw=0.9,
# # )
# # plt.show()

# fig = G.render_surface_stack(
#     mode="auto",            # "grid" for perfect lattices, "auto" is fine
#     z_scale_opinion=4.0,    # taller opinion peaks
#     z_gap_district=1.0,     # district surface above opinion
#     z_gap_social=2.2,       # social layer above district
#     social_lw=0.9,
#     opinion_cmap="RdBu_r"   # nice blue↔red gradient
# )
# plt.show()
