# Novel environment scripts

Scripts for analyzing 3D head position data from the novel environment (DLC-tracked). They compute travel distance in 5-minute bins and generate 3D heatmaps in real-world cage coordinates.

## Requirements

- Python 3
- pandas
- numpy
- matplotlib
- scipy
- seaborn

Install with:

```bash
pip install pandas numpy matplotlib scipy seaborn
```

## Setup

Before running, set the path placeholders in each script to your data and output directories.

### `calculate_3d_travel_distance.py`

At the top of the file, set:

- **`csv_dir`** — directory containing the input CSV files (e.g. `'single_3d_new/day1'`)
- **`out_dir`** — directory where travel-distance outputs will be written (e.g. `'travel_distance'`)

### `generate_3d_heatmap.py`

At the top of the file, set:

- **`output_dir`** — directory where heatmap figures will be saved (e.g. `'3d_heatmaps'`)
- **`csv_dir`** — directory containing the input CSV files (e.g. `'single_3d_new/day1'`)

Replace the literal strings `'<path_to_csv_directory>'` and `'<path_to_output_directory>'` with your paths (as strings, e.g. `'./data/day1'`).

## Input data

- CSVs should contain columns: `rotated_head_x`, `rotated_head_y`, `rotated_head_z` (relative coordinates).
- For CM83, the scripts expect either a pre-built combined file or the two parts `CM83_0822_1_single_3D_filtered_50_rotated_minus_15.csv` and `CM83_0822_2_single_3D_rotated_minus_15.csv` in `csv_dir`; they will create `CM83_0822_combined_single_3D_rotated_minus_15.csv` if missing.
- Cage dimensions (CAGE_X, CAGE_Y, CAGE_Z in cm) are set in both scripts and must match your setup.

## Scripts

### `calculate_3d_travel_distance.py`

Computes 3D travel distance per session in 5-minute chunks (real-world scale, meters), using the same coordinate conversion as the heatmap script.

**Outputs:**

- `travel_distance_5min_WT.csv` — rows = 5-min bins, columns = monkey_day (WT)
- `travel_distance_5min_Shank3.csv` — same for Shank3
- `travel_distance_5min_note.txt` — notes on special handling (e.g. CM78_1031, CM83_0822)
- `travel_distance_5min_boxplot.png` — box plot of travel distance by chunk and genotype

**Run:**

```bash
python calculate_3d_travel_distance.py
```

### `generate_3d_heatmap.py`

Builds 3D heatmaps of head position (density in real-world cage coordinates) for each session and group-averaged plots.

**Outputs:**

- One PNG per session (e.g. `CF52_coolwarm.png`) in `output_dir`
- `Group_Average_3D_heatmap_coolwarm.png` — side-by-side Shank3 vs WT
- `density_cache.npz` — cached density for faster re-runs

**Run:**

```bash
python generate_3d_heatmap.py
```

Camera angle is controlled by `elevation` and `azimuth` at the top of the file; adjust if you want a different view.

## Group definitions

Both scripts use the same genotype groups:

- **Shank3:** CF52, CF55, CF56, CM81, CM83, CM84, CM85  
- **WT:** CF54, CF57, CM78, CM79, CM80, CM82, CM86, CM87  

Edit the `GROUPS` dict in each script if your IDs differ.
