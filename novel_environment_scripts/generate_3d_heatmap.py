import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import gaussian_kde
import os
import glob
import hashlib

output_dir = 'heatmap_output'
os.makedirs(output_dir, exist_ok=True)

elevation = 28.6
azimuth = 116.0

COLORMAP_RELATIVE = 'hot'
COLORMAP_ABSOLUTE = 'hot'
COLORMAP_LOG = 'coolwarm'

CAGE_X = 180.0
CAGE_Y = 150.0
CAGE_Z = 100.0

csv_dir = '<path_to_csv_directory>'

GROUPS = {
    'Shank3': ['CF52', 'CF55', 'CF56', 'CM81', 'CM83', 'CM84', 'CM85'],
    'WT': ['CF54', 'CF57', 'CM78', 'CM79', 'CM80', 'CM82', 'CM86', 'CM87']
}

def find_global_ranges(csv_files):
    """Find min/max ranges across all monkeys with outlier handling."""
    print("Finding global coordinate ranges across all monkeys...")
    
    all_x = []
    all_y = []
    all_z = []
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            x = df['rotated_head_x'].values
            y = df['rotated_head_y'].values
            z = df['rotated_head_z'].values
            
            mask = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
            all_x.extend(x[mask])
            all_y.extend(y[mask])
            all_z.extend(z[mask])
        except Exception as e:
            print(f"Warning: Could not read {csv_file}: {e}")
            continue
    
    all_x = np.array(all_x)
    all_y = np.array(all_y)
    all_z = np.array(all_z)
    
    x_min = np.percentile(all_x, 1)
    x_max = np.percentile(all_x, 99)
    y_min = np.percentile(all_y, 1)
    y_max = np.percentile(all_y, 99)
    z_min = np.percentile(all_z, 1)
    z_max = np.percentile(all_z, 99)
    
    print(f"Relative coordinate ranges (1st-99th percentile):")
    print(f"  X: [{x_min:.3f}, {x_max:.3f}]")
    print(f"  Y: [{y_min:.3f}, {y_max:.3f}]")
    print(f"  Z: [{z_min:.3f}, {z_max:.3f}]")
    
    return {
        'x': (x_min, x_max),
        'y': (y_min, y_max),
        'z': (z_min, z_max)
    }

def convert_to_realworld(x, y, z, rel_ranges):
    """Convert relative coordinates to real-world scale (cm)."""
    x_min_rel, x_max_rel = rel_ranges['x']
    y_min_rel, y_max_rel = rel_ranges['y']
    z_min_rel, z_max_rel = rel_ranges['z']
    
    x_real = ((x - x_min_rel) / (x_max_rel - x_min_rel)) * CAGE_X
    y_real = ((y - y_min_rel) / (y_max_rel - y_min_rel)) * CAGE_Y
    z_real = ((z - z_min_rel) / (z_max_rel - z_min_rel)) * CAGE_Z
    
    return x_real, y_real, z_real

def filter_within_cage(x, y, z):
    """Filter points to only include those within cage dimensions."""
    mask = (x >= 0) & (x <= CAGE_X) & \
           (y >= 0) & (y <= CAGE_Y) & \
           (z >= 0) & (z <= CAGE_Z)
    
    n_before = len(x)
    n_after = mask.sum()
    if n_before != n_after:
        print(f"  Filtered out {n_before - n_after} points outside cage dimensions")
    
    return x[mask], y[mask], z[mask]

def preprocess_cm83_data():
    """Preprocess CM83 data by concatenating two files and filling missing values."""
    file1 = os.path.join(csv_dir, 'CM83_0822_1_single_3D_filtered_50_rotated_minus_15.csv')
    file2 = os.path.join(csv_dir, 'CM83_0822_2_single_3D_rotated_minus_15.csv')
    output_file = os.path.join(csv_dir, 'CM83_0822_combined_single_3D_rotated_minus_15.csv')

    if os.path.exists(output_file):
        print(f"Preprocessed CM83 file already exists: {output_file}")
        print("  Using existing preprocessed file. Delete it to regenerate.")
        return output_file
    
    print("Preprocessing CM83 data...")
    print(f"  Loading {file1}...")
    df1 = pd.read_csv(file1)
    print(f"  Loaded {len(df1)} rows from file 1")
    
    print(f"  Loading {file2}...")
    df2 = pd.read_csv(file2)
    print(f"  Loaded {len(df2)} rows from file 2")
    
    df_combined = pd.concat([df1, df2], ignore_index=True)
    print(f"  Combined data: {len(df_combined)} rows")

    columns_to_fill = ['rotated_head_x', 'rotated_head_y', 'rotated_head_z']

    for col in columns_to_fill:
        if col in df_combined.columns:
            missing_count = df_combined[col].isna().sum()
            if missing_count > 0:
                print(f"  Filling {missing_count} missing values in {col} with moving average...")
                df_combined[col] = df_combined[col].fillna(
                    df_combined[col].rolling(window=5, min_periods=1, center=True).mean()
                )
                df_combined[col] = df_combined[col].ffill().bfill()

    df_combined.to_csv(output_file, index=False)
    print(f"  Saved preprocessed data to: {output_file}")
    
    return output_file

def find_global_log_density_range(csv_files, rel_ranges):
    """Find global log density range across all monkeys for consistent color scaling."""
    print("Calculating global log density range across all monkeys...")
    
    all_log_densities = []
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            x_rel = df['rotated_head_x'].values
            y_rel = df['rotated_head_y'].values
            z_rel = df['rotated_head_z'].values
            
            mask = ~(np.isnan(x_rel) | np.isnan(y_rel) | np.isnan(z_rel))
            x_rel = x_rel[mask]
            y_rel = y_rel[mask]
            z_rel = z_rel[mask]
            
            if len(x_rel) == 0:
                continue

            x, y, z = convert_to_realworld(x_rel, y_rel, z_rel, rel_ranges)
            x, y, z = filter_within_cage(x, y, z)

            if len(x) == 0:
                continue

            positions = np.vstack([x, y, z])
            kde = gaussian_kde(positions)
            density_abs = kde(positions)

            epsilon = 1e-10
            density_log = np.log10(density_abs + epsilon)
            all_log_densities.extend(density_log)
            
        except Exception as e:
            print(f"  Warning: Could not process {csv_file} for global range: {e}")
            continue
    
    if len(all_log_densities) == 0:
        return None, None
    
    all_log_densities = np.array(all_log_densities)
    global_log_min = all_log_densities.min()
    global_log_max = all_log_densities.max()
    
    print(f"  Global log density range: [{global_log_min:.3f}, {global_log_max:.3f}]")
    return global_log_min, global_log_max

def process_csv_file(csv_path, rel_ranges, global_axis_limits):
    """Process a single CSV file and generate a 3D heatmap."""
    filename = os.path.basename(csv_path)
    animal_name = filename.split('_')[0]

    if 'CM83_0822_combined' in filename:
        animal_name = 'CM83'
    
    print(f"\n{'='*60}")
    print(f"Processing: {filename}")
    print(f"{'='*60}")
    
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return False

    x_rel = df['rotated_head_x'].values
    y_rel = df['rotated_head_y'].values
    z_rel = df['rotated_head_z'].values

    mask = ~(np.isnan(x_rel) | np.isnan(y_rel) | np.isnan(z_rel))
    x_rel = x_rel[mask]
    y_rel = y_rel[mask]
    z_rel = z_rel[mask]

    if len(x_rel) == 0:
        print(f"No valid data points in {csv_path}")
        return False

    x, y, z = convert_to_realworld(x_rel, y_rel, z_rel, rel_ranges)
    x, y, z = filter_within_cage(x, y, z)

    if len(x) == 0:
        print(f"No valid data points within cage dimensions in {csv_path}")
        return False

    print(f"Total valid data points (within cage): {len(x)}")
    print(f"X range (cm): [{x.min():.2f}, {x.max():.2f}]")
    print(f"Y range (cm): [{y.min():.2f}, {y.max():.2f}]")
    print(f"Z range (cm): [{z.min():.2f}, {z.max():.2f}]")

    print("Calculating density (log scale)...")
    density_log_min_display = None
    density_log_max_display = None

    try:
        positions = np.vstack([x, y, z])
        kde = gaussian_kde(positions)
        density_abs = kde(positions)
        epsilon = 1e-10
        density_log = np.log10(density_abs + epsilon)
        density_log_min = density_log.min()
        density_log_max = density_log.max()
        density_log_norm = (density_log - density_log_min) / (density_log_max - density_log_min)

        print("Density calculation completed.")
        print(f"  Log density range: [{density_log_min:.3f}, {density_log_max:.3f}]")
        
        colors = density_log_norm
        density_log_min_display = density_log_min
        density_log_max_display = density_log_max
        
    except Exception as e:
        print(f"KDE calculation failed: {e}")
        print("Using z-coordinate for coloring as fallback...")
        z_norm = (z - z.min()) / (z.max() - z.min())
        colors = z_norm

    if len(x) > 50000:
        indices = np.random.choice(len(x), 50000, replace=False)
        x_vis = x[indices]
        y_vis = y[indices]
        z_vis = z[indices]
        colors_vis = colors[indices]
        print(f"Subsampled to {len(x_vis)} points for visualization")
    else:
        x_vis = x
        y_vis = y
        z_vis = z
        colors_vis = colors

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(
        x_vis,
        -z_vis,
        -y_vis,
        c=colors_vis,
        cmap='coolwarm',
        alpha=0.6,
        s=1,
        vmin=0.0,
        vmax=1.0
    )

    cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_label('Relative Density (0-1)', rotation=270, labelpad=20)

    ax.set_xlabel('x (cm)', fontsize=12)
    ax.set_ylabel('z (horizontal depth, cm)', fontsize=12)
    ax.set_zlabel('y (vertical depth, cm)', fontsize=12)
    ax.set_title(f'3D Heatmap of Monkey Head Position - {animal_name} (1 Hour Video)', 
                 fontsize=14, fontweight='bold')

    ax.view_init(elev=elevation, azim=azimuth)

    ax.set_xlim(global_axis_limits['x'])
    ax.set_ylim(global_axis_limits['y'])
    ax.set_zlim(global_axis_limits['z'])

    ax.grid(False)
    ax.xaxis.pane.set_visible(False)
    ax.yaxis.pane.set_visible(False)
    ax.zaxis.pane.set_visible(False)

    output_filename = f'{animal_name}_coolwarm.png'
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"3D heatmap saved to: {output_path}")

    plt.close()
    return True

def load_group_data(group_name, group_monkeys, csv_files, rel_ranges):
    """Load and combine all data for a group."""
    print(f"\nLoading data for {group_name} group...")
    
    all_x = []
    all_y = []
    all_z = []
    processed_monkeys = set()  # Track which monkeys we've already processed
    
    for csv_file in csv_files:
        filename = os.path.basename(csv_file)
        monkey_name = filename.split('_')[0]
        
        if monkey_name == 'CM83':
            if 'combined' not in filename:
                continue

        if monkey_name in group_monkeys:
            if monkey_name in processed_monkeys:
                continue
            
            try:
                df = pd.read_csv(csv_file)
                x_rel = df['rotated_head_x'].values
                y_rel = df['rotated_head_y'].values
                z_rel = df['rotated_head_z'].values

                mask = ~(np.isnan(x_rel) | np.isnan(y_rel) | np.isnan(z_rel))
                x_rel = x_rel[mask]
                y_rel = y_rel[mask]
                z_rel = z_rel[mask]

                x, y, z = convert_to_realworld(x_rel, y_rel, z_rel, rel_ranges)
                x, y, z = filter_within_cage(x, y, z)

                if len(x) > 0:
                    all_x.extend(x)
                    all_y.extend(y)
                    all_z.extend(z)
                    processed_monkeys.add(monkey_name)  # Mark as processed
                
                print(f"  Loaded {len(x)} points from {monkey_name}")
            except Exception as e:
                print(f"  Warning: Could not load {csv_file}: {e}")
                continue
    
    if len(all_x) == 0:
        return None, None, None
    
    all_x = np.array(all_x)
    all_y = np.array(all_y)
    all_z = np.array(all_z)
    
    print(f"  Total combined points: {len(all_x)}")
    return all_x, all_y, all_z

def create_group_heatmap(x, y, z, group_name, global_axis_limits, vmin, vmax, ax, colors=None, cmap='hot'):
    """Create a 3D heatmap for a group on the given axis."""
    if len(x) == 0:
        return None

    if colors is None:
        print(f"  Calculating density for {group_name}...")
        try:
            positions = np.vstack([x, y, z])
            kde = gaussian_kde(positions)
            density = kde(positions)
            density_norm = (density - density.min()) / (density.max() - density.min())
            colors = density_norm
        except Exception as e:
            print(f"  KDE calculation failed: {e}")
            z_norm = (z - z.min()) / (z.max() - z.min())
            colors = z_norm

    if len(x) > 100000:
        indices = np.random.choice(len(x), 100000, replace=False)
        x_vis = x[indices]
        y_vis = y[indices]
        z_vis = z[indices]
        colors_vis = colors[indices]
    else:
        x_vis = x
        y_vis = y
        z_vis = z
        colors_vis = colors

    scatter = ax.scatter(
        x_vis,
        -z_vis,  # z (horizontal depth) -> y axis (flipped)
        -y_vis,  # y (vertical depth) -> z axis (flipped)
        c=colors_vis,
        cmap=cmap,
        alpha=0.6,
        s=1,
        vmin=vmin,
        vmax=vmax
    )

    ax.set_xlabel('x (cm)', fontsize=12)
    ax.set_ylabel('z (horizontal depth, cm)', fontsize=12)
    ax.set_zlabel('y (vertical depth, cm)', fontsize=12)
    ax.set_title(f'{group_name} Group Average', fontsize=14, fontweight='bold')

    ax.view_init(elev=elevation, azim=azimuth)

    ax.set_xlim(global_axis_limits['x'])
    ax.set_ylim(global_axis_limits['y'])
    ax.set_zlim(global_axis_limits['z'])

    ax.grid(False)
    ax.xaxis.pane.set_visible(False)
    ax.yaxis.pane.set_visible(False)
    ax.zaxis.pane.set_visible(False)
    
    return scatter

def get_data_hash(x_shank3, y_shank3, z_shank3, x_wt, y_wt, z_wt):
    """Create a hash of the data to detect changes."""
    data_str = f"{len(x_shank3)}_{len(x_wt)}_{x_shank3.sum():.6f}_{x_wt.sum():.6f}"
    return hashlib.md5(data_str.encode()).hexdigest()[:8]

def save_density_results(x_shank3, y_shank3, z_shank3, x_wt, y_wt, z_wt,
                        density_abs_shank3, density_abs_wt, data_hash):
    """Save density results to file."""
    cache_file = os.path.join(output_dir, 'density_cache.npz')
    np.savez_compressed(
        cache_file,
        x_shank3=x_shank3,
        y_shank3=y_shank3,
        z_shank3=z_shank3,
        x_wt=x_wt,
        y_wt=y_wt,
        z_wt=z_wt,
        density_abs_shank3=density_abs_shank3,
        density_abs_wt=density_abs_wt,
        data_hash=data_hash
    )
    print(f"  Saved density results to: {cache_file}")

def load_density_results(x_shank3, y_shank3, z_shank3, x_wt, y_wt, z_wt):
    """Load density results from file if they exist and match current data."""
    cache_file = os.path.join(output_dir, 'density_cache.npz')
    
    if not os.path.exists(cache_file):
        return None
    
    try:
        print(f"  Checking for cached density results...")
        cache_data = np.load(cache_file)

        cached_hash = str(cache_data['data_hash'])
        current_hash = get_data_hash(x_shank3, y_shank3, z_shank3, x_wt, y_wt, z_wt)

        shapes_match = (
            len(cache_data['x_shank3']) == len(x_shank3) and
            len(cache_data['x_wt']) == len(x_wt)
        )
        
        if cached_hash == current_hash and shapes_match:
            print(f"  Found matching cached density results!")
            return {
                'density_abs_shank3': cache_data['density_abs_shank3'],
                'density_abs_wt': cache_data['density_abs_wt']
            }
        else:
            print(f"  Cached data doesn't match current data (hash or shape mismatch). Recalculating...")
            return None
    except Exception as e:
        print(f"  Error loading cache: {e}. Recalculating...")
        return None

def create_group_average_plots(csv_files, rel_ranges, global_axis_limits):
    """Create side-by-side group average heatmaps in three versions."""
    print(f"\n{'='*60}")
    print("Creating group average heatmaps...")
    print(f"{'='*60}")

    x_shank3, y_shank3, z_shank3 = load_group_data('Shank3', GROUPS['Shank3'], csv_files, rel_ranges)
    x_wt, y_wt, z_wt = load_group_data('WT', GROUPS['WT'], csv_files, rel_ranges)
    
    if x_shank3 is None or x_wt is None:
        print("Error: Could not load data for one or both groups")
        return False

    print("\nChecking for cached density results...")
    cached_results = load_density_results(x_shank3, y_shank3, z_shank3, x_wt, y_wt, z_wt)
    
    if cached_results is not None:
        density_abs_shank3 = cached_results['density_abs_shank3']
        density_abs_wt = cached_results['density_abs_wt']
        print("  Using cached density values")
    else:
        print("\nCalculating absolute density (this may take a while)...")
        all_positions_shank3 = np.vstack([x_shank3, y_shank3, z_shank3])
        all_positions_wt = np.vstack([x_wt, y_wt, z_wt])

        try:
            kde_shank3 = gaussian_kde(all_positions_shank3)
            kde_wt = gaussian_kde(all_positions_wt)
            
            density_abs_shank3 = kde_shank3(all_positions_shank3)
            density_abs_wt = kde_wt(all_positions_wt)

            data_hash = get_data_hash(x_shank3, y_shank3, z_shank3, x_wt, y_wt, z_wt)
            save_density_results(x_shank3, y_shank3, z_shank3, x_wt, y_wt, z_wt,
                                density_abs_shank3, density_abs_wt, data_hash)
        except Exception as e:
            print(f"Error calculating density: {e}")
            return False

    try:
        density_all_abs = np.concatenate([density_abs_shank3, density_abs_wt])
        density_min_abs = density_all_abs.min()
        density_max_abs = density_all_abs.max()

        density_norm_shank3 = (density_abs_shank3 - density_min_abs) / (density_max_abs - density_min_abs)
        density_norm_wt = (density_abs_wt - density_min_abs) / (density_max_abs - density_min_abs)

        epsilon = 1e-10
        density_log_shank3 = np.log10(density_abs_shank3 + epsilon)
        density_log_wt = np.log10(density_abs_wt + epsilon)
        density_log_all = np.concatenate([density_log_shank3, density_log_wt])
        density_log_min = density_log_all.min()
        density_log_max = density_log_all.max()
        density_log_norm_shank3 = (density_log_shank3 - density_log_min) / (density_log_max - density_log_min)
        density_log_norm_wt = (density_log_wt - density_log_min) / (density_log_max - density_log_min)
        
        print(f"  Absolute density range: [{density_min_abs:.6e}, {density_max_abs:.6e}] cm⁻³")
        print(f"  Log density range: [{density_log_min:.3f}, {density_log_max:.3f}]")
        
    except Exception as e:
        print(f"Error: Could not process density values: {e}")
        return False

    print("\nCreating Group Average: Relative density with coolwarm colormap...")
    fig = plt.figure(figsize=(28, 10))
    ax1 = fig.add_subplot(121, projection='3d')
    scatter1 = create_group_heatmap(x_shank3, y_shank3, z_shank3, 'Shank3', 
                                     global_axis_limits, 0.0, 1.0, ax1, 
                                     colors=density_norm_shank3, cmap='coolwarm')
    ax2 = fig.add_subplot(122, projection='3d')
    scatter2 = create_group_heatmap(x_wt, y_wt, z_wt, 'WT', 
                                     global_axis_limits, 0.0, 1.0, ax2,
                                     colors=density_norm_wt, cmap='coolwarm')
    if scatter1 is not None:
        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.87, 0.15, 0.02, 0.7])
        cbar = plt.colorbar(scatter1, cax=cbar_ax)
        cbar.set_label('Relative Density (0-1)', rotation=270, labelpad=20)
    output_path = os.path.join(output_dir, 'Group_Average_3D_heatmap_coolwarm.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()
    
    print(f"\nGroup average heatmap created successfully!")
    return True

if __name__ == "__main__":
    cm83_preprocessed = preprocess_cm83_data()

    csv_pattern = os.path.join(csv_dir, '*.csv')
    csv_files = sorted(glob.glob(csv_pattern))

    csv_files = [f for f in csv_files if 'CM83_0822_1' not in f and 'CM83_0822_2' not in f]
    if os.path.exists(cm83_preprocessed):
        if cm83_preprocessed not in csv_files:
            csv_files.append(cm83_preprocessed)
        csv_files = sorted(csv_files)

    csv_files = list(dict.fromkeys(csv_files))

    if not csv_files:
        print(f"No CSV files found in {csv_dir}")
    else:
        print(f"Found {len(csv_files)} CSV files to process")
        print(f"Using elevation={elevation}°, azimuth={azimuth}°\n")

        rel_ranges = find_global_ranges(csv_files)

        x_min_rel, x_max_rel = rel_ranges['x']
        y_min_rel, y_max_rel = rel_ranges['y']
        z_min_rel, z_max_rel = rel_ranges['z']

        x_min_real = 0.0
        x_max_real = CAGE_X
        y_min_real = 0.0
        y_max_real = CAGE_Y
        z_min_real = 0.0
        z_max_real = CAGE_Z

        y_axis_min = -z_max_real
        y_axis_max = -z_min_real
        z_axis_min = -y_max_real
        z_axis_max = -y_min_real

        padding = 0.05
        x_range = x_max_real - x_min_real
        y_range = y_axis_max - y_axis_min
        z_range = z_axis_max - z_axis_min
        
        global_axis_limits = {
            'x': (x_min_real - x_range * padding, x_max_real + x_range * padding),
            'y': (y_axis_min - y_range * padding, y_axis_max + y_range * padding),
            'z': (z_axis_min - z_range * padding, z_axis_max + z_range * padding)
        }
        
        print(f"\nGlobal axis limits (real-world, cm):")
        print(f"  X: [{global_axis_limits['x'][0]:.2f}, {global_axis_limits['x'][1]:.2f}]")
        print(f"  Y (shows -z): [{global_axis_limits['y'][0]:.2f}, {global_axis_limits['y'][1]:.2f}]")
        print(f"  Z (shows -y): [{global_axis_limits['z'][0]:.2f}, {global_axis_limits['z'][1]:.2f}]\n")

        print("\nGenerating individual monkey heatmaps...")
        successful = 0
        failed = 0
        
        for csv_file in csv_files:
            if process_csv_file(csv_file, rel_ranges, global_axis_limits):
                successful += 1
            else:
                failed += 1
        
        print(f"\nIndividual plots: {successful} successful, {failed} failed")

        create_group_average_plots(csv_files, rel_ranges, global_axis_limits)
        
        print(f"\n{'='*60}")
        print(f"Processing complete!")
        print(f"{'='*60}")

