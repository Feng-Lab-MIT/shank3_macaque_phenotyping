"""
Compute 3D travel distance per monkey_day in 5-minute chunks (real-world scale).
Uses the same coordinate conversion pipeline as generate_3d_heatmap.py.
Outputs: WT and Shank3 summary CSVs (rows = 5-min bins, columns = monkey_day, values in meters)
and a binned box plot (x = chunk, y = travel distance m, one box per genotype per bin).
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Real-world cage dimensions (cm) - must match generate_3d_heatmap.py
CAGE_X = 180.0
CAGE_Y = 150.0
CAGE_Z = 100.0

csv_dir = '<path_to_csv_directory>'
out_dir = 'travel_distance_output'
os.makedirs(out_dir, exist_ok=True)

FPS = 24
SEC_PER_CHUNK = 300
FRAMES_PER_CHUNK = FPS * SEC_PER_CHUNK
N_CHUNKS = 12
BIN_LABELS = [f'[{5*k},{5*(k+1)})' for k in range(11)] + ['[55,60]']

GROUPS = {
    'Shank3': ['CF52', 'CF55', 'CF56', 'CM81', 'CM83', 'CM84', 'CM85'],
    'WT': ['CF54', 'CF57', 'CM78', 'CM79', 'CM80', 'CM82', 'CM86', 'CM87']
}


def find_global_ranges(csv_files):
    """Find min/max coordinate ranges across all files (1st–99th percentile)."""
    all_x, all_y, all_z = [], [], []
    for path in csv_files:
        try:
            df = pd.read_csv(path)
            x = df['rotated_head_x'].values
            y = df['rotated_head_y'].values
            z = df['rotated_head_z'].values
            mask = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
            all_x.extend(x[mask])
            all_y.extend(y[mask])
            all_z.extend(z[mask])
        except Exception as e:
            print(f"Warning: skip {path}: {e}")
    all_x = np.array(all_x)
    all_y = np.array(all_y)
    all_z = np.array(all_z)
    return {
        'x': (np.percentile(all_x, 1), np.percentile(all_x, 99)),
        'y': (np.percentile(all_y, 1), np.percentile(all_y, 99)),
        'z': (np.percentile(all_z, 1), np.percentile(all_z, 99)),
    }


def convert_to_realworld(x, y, z, rel_ranges):
    """Convert relative coordinates to real-world cm."""
    x_min, x_max = rel_ranges['x']
    y_min, y_max = rel_ranges['y']
    z_min, z_max = rel_ranges['z']
    x_real = ((x - x_min) / (x_max - x_min)) * CAGE_X
    y_real = ((y - y_min) / (y_max - y_min)) * CAGE_Y
    z_real = ((z - z_min) / (z_max - z_min)) * CAGE_Z
    return x_real, y_real, z_real


def within_cage(x, y, z):
    """Boolean mask: True if point is inside cage."""
    return (
        (x >= 0) & (x <= CAGE_X) &
        (y >= 0) & (y <= CAGE_Y) &
        (z >= 0) & (z <= CAGE_Z)
    )


def preprocess_cm83_data():
    """Build combined CM83 file if missing (same as generate_3d_heatmap.py)."""
    file1 = os.path.join(csv_dir, 'CM83_0822_1_single_3D_filtered_50_rotated_minus_15.csv')
    file2 = os.path.join(csv_dir, 'CM83_0822_2_single_3D_rotated_minus_15.csv')
    output_file = os.path.join(csv_dir, 'CM83_0822_combined_single_3D_rotated_minus_15.csv')
    if os.path.exists(output_file):
        return output_file
    if not os.path.exists(file1) or not os.path.exists(file2):
        return output_file
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    df = pd.concat([df1, df2], ignore_index=True)
    for col in ['rotated_head_x', 'rotated_head_y', 'rotated_head_z']:
        if col in df.columns:
            df[col] = df[col].fillna(
                df[col].rolling(window=5, min_periods=1, center=True).mean()
            ).ffill().bfill()
    df.to_csv(output_file, index=False)
    return output_file


def monkey_day_from_path(csv_path):
    """e.g. CF52_0822_single_3D_... -> CF52_0822; CM83_0822_combined_... -> CM83_0822."""
    base = os.path.basename(csv_path).replace('.csv', '')
    parts = base.split('_')
    if len(parts) >= 2:
        return '_'.join(parts[:2])
    return base


def genotype_from_monkey(monkey_day):
    """Return 'WT' or 'Shank3' from monkey_day prefix (e.g. CF52_0822 -> Shank3)."""
    monkey = monkey_day.split('_')[0]
    for g, names in GROUPS.items():
        if monkey in names:
            return g
    return None


def travel_distance_per_chunk(csv_path, rel_ranges):
    """
    For one CSV: convert to real-world, then for each 5-min chunk compute
    sum of consecutive-frame distances (only segments with both endpoints inside cage).
    Returns: 1D array of length N_CHUNKS, in meters.
    """
    df = pd.read_csv(csv_path)
    x = df['rotated_head_x'].values.astype(float)
    y = df['rotated_head_y'].values.astype(float)
    z = df['rotated_head_z'].values.astype(float)
    n = len(x)
    x_r, y_r, z_r = convert_to_realworld(x, y, z, rel_ranges)
    inside = within_cage(x_r, y_r, z_r)
    seg_inside = inside[:-1] & inside[1:]
    dx = np.diff(x_r)
    dy = np.diff(y_r)
    dz = np.diff(z_r)
    dist_cm = np.sqrt(dx*dx + dy*dy + dz*dz)
    dist_cm[~seg_inside] = 0.0
    chunk_dist_cm = np.zeros(N_CHUNKS)
    for i in range(min(n - 1, (N_CHUNKS * FRAMES_PER_CHUNK) - 1)):
        ch = i // FRAMES_PER_CHUNK
        if ch < N_CHUNKS:
            chunk_dist_cm[ch] += dist_cm[i]
    return chunk_dist_cm / 100.0


def fill_missing_chunks(monkey_day, arr):
    """
    Apply special-case fills for sessions with truncated or gapped video.
    - CM78_1031: video cut at 55 min → [55,60] filled with [50,55) value.
    - CM83_0822: first part 0–20 min, second 25–55 min → (20,25] and [55,60] filled
      as: (20,25] = mean of [15,20) and [25,30); [55,60] = [50,55) value.
    Modifies arr in place; returns arr.
    """
    if monkey_day == 'CM78_1031':
        arr[11] = arr[10]
    elif monkey_day == 'CM83_0822':
        arr[4] = (arr[3] + arr[5]) / 2.0
        arr[11] = arr[10]
    return arr


def main():
    preprocess_cm83_data()
    csv_pattern = os.path.join(csv_dir, '*.csv')
    csv_files = sorted(glob.glob(csv_pattern))
    csv_files = [f for f in csv_files if 'CM83_0822_1' not in f and 'CM83_0822_2' not in f]
    cm83_combined = os.path.join(csv_dir, 'CM83_0822_combined_single_3D_rotated_minus_15.csv')
    if os.path.exists(cm83_combined) and cm83_combined not in csv_files:
        csv_files.append(cm83_combined)
    csv_files = sorted(list(dict.fromkeys(csv_files)))

    if not csv_files:
        print("No CSV files found.")
        return

    rel_ranges = find_global_ranges(csv_files)
    print("Global ranges found.")

    results = {}
    for path in csv_files:
        md = monkey_day_from_path(path)
        try:
            results[md] = travel_distance_per_chunk(path, rel_ranges)
            fill_missing_chunks(md, results[md])
        except Exception as e:
            print(f"Skip {path}: {e}")

    wt_cols = [md for md in results if genotype_from_monkey(md) == 'WT']
    shank3_cols = [md for md in results if genotype_from_monkey(md) == 'Shank3']
    df_wt = pd.DataFrame(
        {md: results[md] for md in wt_cols},
        index=BIN_LABELS
    )
    df_shank3 = pd.DataFrame(
        {md: results[md] for md in shank3_cols},
        index=BIN_LABELS
    )

    df_wt.to_csv(os.path.join(out_dir, 'travel_distance_5min_WT.csv'))
    df_shank3.to_csv(os.path.join(out_dir, 'travel_distance_5min_Shank3.csv'))
    print(f"Saved {out_dir}/travel_distance_5min_WT.csv and travel_distance_5min_Shank3.csv")

    note_path = os.path.join(out_dir, 'travel_distance_5min_note.txt')
    with open(note_path, 'w') as f:
        f.write("Travel distance 5-min summary: special processing applied\n")
        f.write("=" * 60 + "\n\n")
        f.write("1. CM78_1031\n")
        f.write("   The video was cut at 55 minutes due to technical issues.\n")
        f.write("   The [55,60] min chunk is filled with the [50,55) value.\n\n")
        f.write("2. CM83_0822\n")
        f.write("   The video was corrupted; only two parts could be recovered:\n")
        f.write("   - Part 1: 0–20 min\n")
        f.write("   - Part 2: 25–55 min\n")
        f.write("   The missing chunks are filled as follows:\n")
        f.write("   - (20,25] min: average of [15,20) and [25,30)\n")
        f.write("   - [55,60] min: value from [50,55)\n")
    print(f"Saved {note_path}")

    rows = []
    for md, arr in results.items():
        g = genotype_from_monkey(md)
        if g is None:
            continue
        for k, label in enumerate(BIN_LABELS):
            rows.append({'bin': label, 'genotype': g, 'travel_distance_m': arr[k]})
    df_long = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.boxplot(
        data=df_long, x='bin', y='travel_distance_m', hue='genotype',
        order=BIN_LABELS, ax=ax
    )
    ax.set_xlabel('5-min chunk')
    ax.set_ylabel('Travel distance (m)')
    ax.set_title('Travel distance by 5-min chunk and genotype')
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'travel_distance_5min_boxplot.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_dir}/travel_distance_5min_boxplot.png")


if __name__ == '__main__':
    main()
