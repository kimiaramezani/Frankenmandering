# Run this script to generate the grid and partition files

python make_grid.py --config grid_config.yaml --write_seeds

# For runB you can use (As it is not set in the seed part of the 
# grid_config.yaml file)
python make_grid.py --config grid_config.yaml --write_seeds --seeds-key "12x12_K6:runB"

# If you want to override grid params from the CLI (CLI beats config)
python make_grid.py --config grid_config.yaml --write_seeds --H 14 --W 14 --K 7

# Only write nodes/edges (no seeds partition)
python make_grid.py --config grid_config.yaml

# What the outputs will look like
# Under io.outdir (e.g., data/):
# grid_{W}x{H}__K{K}__adj4__wrap0__seed{seed}__v{YYMMDD}__{HHMMSS}__seeds-<run>__nodes.csv
# grid_{W}x{H}__K{K}__adj4__wrap0__seed{seed}__v{YYMMDD}__{HHMMSS}__seeds-<run>__edges_grid.csv
# grid_{W}x{H}__K{K}__adj4__wrap0__seed{seed}__v{YYMMDD}__{HHMMSS}__seeds-<run>__partition_init.csv


# To Visualize the created grids (separately)
python grid_graphs.py --base "grid_12x12__K6__adj4__wrap0__seed42__v250904__151046__seeds-runA" --in-dir data --out-dir viz

(--out-dir output_directory where you want to store the viz)
