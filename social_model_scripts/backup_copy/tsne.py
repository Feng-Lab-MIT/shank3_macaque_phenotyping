import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
import seaborn as sns
import matplotlib.colors as mcolors
from tqdm import tqdm
from openTSNE import TSNEEmbedding, affinity, initialization

def compute_tsne_in_batches_uniform(data, n_batches, perplexity=500, n_iter=2000, early_exaggeration=12, exaggeration=1, random_seed=42, early_exaggeration_iter=250):
    """Compute t-SNE in batches for uniform sampling"""
    # Calculate the total rows
    total_rows = data.shape[0]
    
    # Make sure total_rows is divisible by n_batches by removing excess frames
    frames_to_remove = total_rows % n_batches
    if frames_to_remove > 0:
        print(f"Removing {frames_to_remove} frames to make data divisible by {n_batches}")
        data = data[:-frames_to_remove]
        total_rows = data.shape[0]
        print(f"Adjusted total rows: {total_rows}")

    # Initialize the affinities variable
    affinities = None

    all_embeddings = []

    # Process each batch
    for i in tqdm(range(n_batches), desc="Processing batches"):
        batch_indices = np.arange(i, total_rows, n_batches)
        batch_data = data[batch_indices]

        if affinities is None:
            # Calculate affinities for the first batch
            affinities = affinity.PerplexityBasedNN(
                batch_data,
                perplexity=perplexity,
                metric="euclidean",
                random_state=random_seed,
                n_jobs=-1,
                verbose=True,
            )

        # Generate initial coordinates using PCA
        init = initialization.pca(batch_data, random_state=random_seed)

        # Construct the TSNEEmbedding object for the batch
        embedding = TSNEEmbedding(
            init,
            affinities,
            negative_gradient_method="fft",
            n_jobs=-1,
            verbose=True,
        )

        # Optimize the embedding with early exaggeration phase
        embedding = embedding.optimize(n_iter=early_exaggeration_iter, exaggeration=early_exaggeration)

        # Regular optimization phase
        embedding = embedding.optimize(n_iter=(n_iter-early_exaggeration_iter), exaggeration=exaggeration)

        # Store the embedding
        all_embeddings.append(embedding.view(np.ndarray))

    # Combine embeddings from all batches
    combined_embeddings = np.vstack(all_embeddings)

    return combined_embeddings

def restore_tsne_order_subset(pca_vectors, tsne_results, n_batches, subset_indices):
    """
    Restore the order of tsne_results to match the original order of pca_vectors for subset data.
    """
    # Make sure pca_vectors is divisible by n_batches (same adjustment as in tSNE function)
    total_rows = len(pca_vectors)
    frames_to_remove = total_rows % n_batches
    if frames_to_remove > 0:
        pca_vectors = pca_vectors[:-frames_to_remove]
        subset_indices = subset_indices[:-frames_to_remove]
        total_rows = len(pca_vectors)
    
    # Process batches again to get the correct indices
    frame_indices_subset = []
    for i in tqdm(range(n_batches), desc="Processing batches for order restoration"):
        batch_indices = np.arange(i, total_rows, n_batches)
        frame_indices_subset.extend(batch_indices)
    
    # Create a DataFrame with the restored order
    tsne_df = pd.DataFrame(tsne_results, columns=['TSNE1', 'TSNE2'])
    
    # Add subset indices to the DataFrame
    tsne_df['subset_index'] = frame_indices_subset
    
    return tsne_df

def perform_agglomerative_clustering(data, n_clusters=32, n_neighbors=50):
    """Perform agglomerative clustering with connectivity constraints"""
    connectivity = kneighbors_graph(data, n_neighbors=n_neighbors, include_self=False)
    agg_clustering = AgglomerativeClustering(n_clusters=n_clusters, connectivity=connectivity, linkage='ward')
    labels = agg_clustering.fit_predict(data)
    return labels

def plot_tsne_with_labels_subset(tsne_df, n_clusters, folder, perplexity, exaggeration, n_neighbors, category_name, sequence_length):
    """Plot t-SNE results with cluster labels for subset data"""
    sorted_clusters = sorted(tsne_df['Cluster'].unique())
    color_palette = sns.color_palette("husl", n_clusters)
    cluster_colors = {cluster: color for cluster, color in zip(sorted_clusters, color_palette)}
    
    # Create a discrete colormap
    cmap = mcolors.ListedColormap(color_palette)
    bounds = np.arange(len(sorted_clusters) + 1)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    
    # Create the plot with cluster labels
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(tsne_df['TSNE1'], tsne_df['TSNE2'], c=tsne_df['Cluster'].astype(int), cmap=cmap, norm=norm, s=0.5, marker='o', edgecolors='none')
    
    # Add title and labels
    ax.set_title(f't-SNE of Latent Space Representations ({category_name}) with Agglomerative Clustering')
    ax.set_xlabel('t-SNE Component 1')
    ax.set_ylabel('t-SNE Component 2')
    
    # Add cluster labels
    for cluster in sorted_clusters:
        cluster_points = tsne_df[tsne_df['Cluster'] == cluster]
        x_mean = cluster_points['TSNE1'].mean()
        y_mean = cluster_points['TSNE2'].mean()
        ax.text(x_mean, y_mean, str(cluster), fontsize=14, weight='bold')
    
    # Add color bar
    sorted_clusters = sorted([int(cluster) for cluster in sorted_clusters])
    cmap = mcolors.ListedColormap([cluster_colors[str(cluster)] for cluster in sorted_clusters])
    cbar = plt.colorbar(scatter, ticks=bounds + 0.5)
    cbar.set_label('Cluster Label')
    cbar.set_ticks(bounds[:-1] + 0.5)
    cbar.set_ticklabels(sorted_clusters)
    
    # Save the plot
    plt.savefig(f'{folder}cluster_plot_subset_{category_name.replace(" ", "_")}_n_clusters_{n_clusters}_exaggeration_{exaggeration}_perplexity_{perplexity}_n_neighbors_{n_neighbors}_no_velocity_test_depth_sequence_length_{sequence_length}.png', dpi=300)
    plt.close('all')

def load_frame_indices(output_dir):
    """Load frame indices from latent extraction output"""
    frame_indices = []
    
    indices_files = sorted([f for f in os.listdir(output_dir) if f.startswith('indices_') and f.endswith('.npy')])
    
    for filename in tqdm(indices_files, desc="Loading Frame Indices"):
        indices_data = np.load(os.path.join(output_dir, filename))
        frame_indices.append(indices_data)
    
    return np.concatenate(frame_indices, axis=0)

# Configuration - these can be updated by the pipeline script
category_name = 'Bottom_Playing_and_Alone'

# Model parameters - change these to match your PCA script
sequence_length = 240
latent_dim = 64
bottleneck_dim = 32

# t-SNE and clustering parameters - can be updated by pipeline
n_batches = 6
n_clusters = 32
perplexity = 50
desired_categories = ['Bottom Playing', 'Bottom Alone']

# Folder paths - can be updated by pipeline
pca_input_folder = '/media/fenglab/newssd/social/'
latent_input_folder = '/media/fenglab/newssd/social/'
tsne_output_folder = '/media/fenglab/newssd/social/'

def main():
    """Main function for t-SNE analysis"""
    # Construct paths based on current parameters and provided folders
    base_folder = f'{pca_input_folder}bottom_playing_and_alone_pca_seq_{sequence_length}_latent_{latent_dim}_bottleneck_{bottleneck_dim}_no_velocity_test_depth/'
    pca_results_folder = os.path.join(base_folder, 'pca_results')
    latent_dir = f'{latent_input_folder}bottom_playing_and_alone_latent_seq_{sequence_length}_latent_{latent_dim}_bottleneck_{bottleneck_dim}_no_velocity_test_depth/'
    output_folder = f'{tsne_output_folder}bottom_playing_and_alone_tsne_seq_{sequence_length}_latent_{latent_dim}_bottleneck_{bottleneck_dim}_no_velocity_test_depth/'
    
    # Find the PCA vectors file and extract n_components from filename
    pca_vectors_file = None
    n_components = None
    
    if os.path.exists(pca_results_folder):
        # Look for PCA vectors files
        pca_files = [f for f in os.listdir(pca_results_folder) if f.startswith(f'pca_vectors_{category_name.replace(" ", "_")}_n_components_') and f.endswith('.npy')]
        
        if not pca_files:
            print(f"Error: No PCA vectors files found in {pca_results_folder}")
            print("Available files in pca_results folder:")
            for f in os.listdir(pca_results_folder):
                print(f"  {f}")
            return
        
        # Use the first (and likely only) PCA vectors file
        pca_vectors_file = os.path.join(pca_results_folder, pca_files[0])
        print(f"Found PCA vectors file: {pca_files[0]}")
        
        # Extract n_components from filename
        import re
        match = re.search(r'n_components_(\d+)_no_velocity_test_depth_sequence_length_\d+\.npy', pca_files[0])
        if match:
            n_components = int(match.group(1))
            print(f"Auto-detected n_components: {n_components}")
        else:
            print(f"Error: Could not extract n_components from filename: {pca_files[0]}")
            return
    else:
        print(f"Error: PCA results folder not found at {pca_results_folder}")
        return
    
    # Load the saved PCA vectors
    if not os.path.exists(pca_vectors_file):
        print(f"Error: PCA vectors file not found at {pca_vectors_file}")
        return
    
    pca_vectors_subset = np.load(pca_vectors_file)
    print(f"Loaded PCA vectors with shape: {pca_vectors_subset.shape}")
    
    # Load the original frame indices
    frame_indices = load_frame_indices(latent_dir)
    print(f"Loaded original frame indices with shape: {frame_indices.shape}")
    print(f"Type of frame_indices: {type(frame_indices)}")
    print(f"First few elements: {frame_indices[:3]}")
    print(f"Type of first element: {type(frame_indices[0])}")
    if len(frame_indices) > 0:
        print(f"Shape of first element: {frame_indices[0].shape if hasattr(frame_indices[0], 'shape') else 'No shape'}")
    
    # Generate the di_mp_dic mapping
    pre_sample_feature = pd.read_csv('/media/fenglab/New Volume/River_data_backup/2nd_cohort_social/3d/features/pre/probe_z_calib_0511.csv', index_col=[0], header=[0])
    pre_monkey_day_list = [col for col in pre_sample_feature.columns if '_' in col]
    post_sample_feature = pd.read_csv('/media/fenglab/New Volume/River_data_backup/2nd_cohort_social/3d/features/post/probe_z_calib_202309.csv', index_col=[0], header=[0])
    post_monkey_day_list = [col for col in post_sample_feature.columns if '_' in col]
    
    monkey_day_list = pre_monkey_day_list + post_monkey_day_list
    monkey_day_list.sort()
    
    # Create the mapping dictionary
    di_mp_dic = {i: s for i, s in enumerate(monkey_day_list)}
    print(f"Generated di_mp_dic with {len(di_mp_dic)} entries")
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Set folder for output
    folder = output_folder
    
    # Run t-SNE and clustering on the subset
    # Use module-level perplexity (can be updated by pipeline)
    for exaggeration in [1]:
        for early_exaggeration_iter in [300]:
            print(f'Perplexity={perplexity}, exaggeration={exaggeration}')
            
            # Check if we can load pre-computed t-SNE results
            tsne_save_file = f'{folder}tsne_df_before_mapping_{category_name.replace(" ", "_")}_perplexity_{perplexity}_exaggeration_{exaggeration}_earlyiter_{early_exaggeration_iter}_pca_{n_components}_n_batches_{n_batches}_no_velocity_test_depth_sequence_length_{sequence_length}.csv'
            
            if os.path.exists(tsne_save_file):
                print(f"Loading pre-computed t-SNE results from: {tsne_save_file}")
                tsne_df = pd.read_csv(tsne_save_file)
                # Convert original_latent_index back to numpy arrays
                import ast
                import re
                print(f"Sample original_latent_index from CSV: {tsne_df['original_latent_index'].iloc[0]}")
                print(f"Type: {type(tsne_df['original_latent_index'].iloc[0])}")
                
                # Handle the string format - it looks like "[0 0]" so we need to add commas
                def convert_to_array(x):
                    # If x is already an integer, return it as a single-element array
                    if isinstance(x, (int, np.integer)):
                        return np.array([x])
                    # If x is a string, parse it
                    elif isinstance(x, str):
                        # Convert "[0 0]" to "[0, 0]" format
                        x_clean = re.sub(r'(\d+)\s+(\d+)', r'\1, \2', x)
                        return np.array(ast.literal_eval(x_clean))
                    else:
                        # If it's already an array, return it
                        return np.array(x)
                
                tsne_df['original_latent_index'] = tsne_df['original_latent_index'].apply(convert_to_array)
                print(f"Loaded tsne_df with shape: {tsne_df.shape}")
                print(f"Columns in loaded tsne_df: {tsne_df.columns.tolist()}")
                
                # Extract t-SNE coordinates for clustering - check which columns exist
                if 'TSNE1' in tsne_df.columns and 'TSNE2' in tsne_df.columns:
                    tsne_results = tsne_df[['TSNE1', 'TSNE2']].values
                else:
                    # If t-SNE columns don't exist, we need to recompute t-SNE
                    print("t-SNE columns not found in saved file, recomputing t-SNE...")
                    tsne_results = compute_tsne_in_batches_uniform(
                        pca_vectors_subset, 
                        n_batches=n_batches, 
                        perplexity=perplexity, 
                        n_iter=1000, 
                        early_exaggeration=12, 
                        exaggeration=exaggeration, 
                        random_seed=42, 
                        early_exaggeration_iter=early_exaggeration_iter
                    )
                    
                    # Restore order for subset data
                    # Create sequential indices for the function (it just needs the length)
                    sequential_indices = np.arange(len(pca_vectors_subset))
                    tsne_df = restore_tsne_order_subset(pca_vectors_subset, tsne_results, n_batches, sequential_indices)
                    
                    # Add original frame information - use frame_indices which contains (DataFrame Index, Frame Index) tuples
                    tsne_df['original_latent_index'] = [frame_indices[i] for i in tsne_df['subset_index']]
            else:
                print("Computing t-SNE from scratch...")
                # Compute t-SNE on subset data
                tsne_results = compute_tsne_in_batches_uniform(
                    pca_vectors_subset, 
                    n_batches=n_batches, 
                    perplexity=perplexity, 
                    n_iter=1000, 
                    early_exaggeration=12, 
                    exaggeration=exaggeration, 
                    random_seed=42, 
                    early_exaggeration_iter=early_exaggeration_iter
                )
                
                # Restore order for subset data
                # Create sequential indices for the function (it just needs the length)
                sequential_indices = np.arange(len(pca_vectors_subset))
                tsne_df = restore_tsne_order_subset(pca_vectors_subset, tsne_results, n_batches, sequential_indices)
                
                # Add original frame information - use frame_indices which contains (DataFrame Index, Frame Index) tuples
                tsne_df['original_latent_index'] = [frame_indices[i] for i in tsne_df['subset_index']]
                
                # Save tsne_df before mapping to avoid re-running t-SNE
                tsne_df.to_csv(tsne_save_file, index=False)
                print(f"Saved tsne_df before mapping to: {tsne_save_file}")
            
            # Map back to DataFrame Index and Frame Index
            print(f"Sample original_latent_index values: {tsne_df['original_latent_index'].head()}")
            print(f"Sample frame_indices access: {frame_indices[tsne_df['original_latent_index'].iloc[0]]}")
            print(f"Type of accessed element: {type(frame_indices[tsne_df['original_latent_index'].iloc[0]])}")
            
            # Debug: Check the actual structure of frame_indices
            print(f"Sample frame_indices values: {frame_indices[:5]}")
            print(f"Sample frame_indices type: {type(frame_indices[0])}")
            if hasattr(frame_indices[0], '__len__'):
                print(f"Sample frame_indices length: {len(frame_indices[0])}")
            
            # Handle different data types for original_latent_index
            if hasattr(tsne_df['original_latent_index'].iloc[0], '__len__') and len(tsne_df['original_latent_index'].iloc[0]) == 2:
                # If it's a tuple/array with DataFrame Index and Frame Index
                tsne_df['DataFrame Index'] = [int(i[0]) for i in tsne_df['original_latent_index']]
                tsne_df['Frame Index'] = [int(i[1]) for i in tsne_df['original_latent_index']]
            else:
                # If it's a single integer, it represents the subset_index
                # Check if frame_indices contains tuples or scalars
                if hasattr(frame_indices[0], '__len__') and len(frame_indices[0]) == 2:
                    # frame_indices contains tuples (DataFrame Index, Frame Index)
                    tsne_df['DataFrame Index'] = [int(frame_indices[i[0]][0]) for i in tsne_df['original_latent_index']]
                    tsne_df['Frame Index'] = [int(frame_indices[i[0]][1]) for i in tsne_df['original_latent_index']]
                else:
                    # frame_indices contains scalars - use them as DataFrame Index, Frame Index = 0
                    tsne_df['DataFrame Index'] = [int(frame_indices[i[0]]) for i in tsne_df['original_latent_index']]
                    tsne_df['Frame Index'] = [0] * len(tsne_df)  # Default to 0 if not available
            tsne_df['Monkey_Day'] = tsne_df['DataFrame Index'].map(di_mp_dic)
            
            # Add category information
            tsne_df['Category'] = category_name
            
            # Perform clustering
            for n_neighbors in [50]:
                print(f"Performing agglomerative clustering with n_neighbors = {n_neighbors}")
                labels = perform_agglomerative_clustering(tsne_results, n_clusters=n_clusters, n_neighbors=n_neighbors)
                
                # Add labels to the DataFrame
                tsne_df['Cluster'] = labels.astype(str)
                
                # Plot results
                plot_tsne_with_labels_subset(
                    tsne_df, 
                    n_clusters, 
                    folder, 
                    perplexity, 
                    exaggeration, 
                    n_neighbors, 
                    category_name,
                    sequence_length
                )
                
                # Save results
                file_name = f'{folder}tsne_results_subset_{category_name.replace(" ", "_")}_perplexity_{perplexity}_exaggeration_{exaggeration}_earlyiter_{early_exaggeration_iter}_pca_{n_components}_n_neighbors_{n_neighbors}_n_batches_{n_batches}_n_clusters_{n_clusters}_no_velocity_test_depth_sequence_length_{sequence_length}.csv'
                tsne_df.to_csv(file_name, index=False)
                print(f"Results saved to: {file_name}")
    
    print("✅ Standalone tSNE analysis complete!")

if __name__ == "__main__":
    main()
