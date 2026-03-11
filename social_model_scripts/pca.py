import pandas as pd
import numpy as np
import os
import pickle
from sklearn.decomposition import PCA
from tqdm import tqdm

# Configuration - these can be updated by the pipeline script
# Category folder name for paths (set by pipeline when using config; default for standalone)
category_folder_name = 'all_categories'

# Model parameters - change these to match your latent extraction
sequence_length = 240
latent_dim = 64
bottleneck_dim = 32

# Folder paths - can be updated by pipeline
latent_input_folder = '/path/to/latent_output/'
pca_output_folder = '/path/to/pca_output/'

def load_frame_indices(latent_dir):
    """Load frame indices from the subset latent extraction"""
    frame_indices = []
    
    indices_files = sorted([f for f in os.listdir(latent_dir) if f.startswith('indices_') and f.endswith('.npy')])
    
    for filename in tqdm(indices_files, desc="Loading Frame Indices"):
        indices_data = np.load(os.path.join(latent_dir, filename))
        frame_indices.append(indices_data)
    
    return np.concatenate(frame_indices, axis=0)

def load_latent_representations_batch(latent_dir, batch_size=10000):
    """Load latent representations in batches to avoid memory issues"""
    latent_files = sorted([f for f in os.listdir(latent_dir) if f.startswith('latent_representations_')])
    
    print(f"Found {len(latent_files)} latent representation files")
    
    # First, get the shape of the data
    first_file = os.path.join(latent_dir, latent_files[0])
    first_data = np.load(first_file)
    feature_dim = first_data.shape[1]
    
    # Calculate total samples
    total_samples = 0
    for filename in tqdm(latent_files, desc="Counting samples"):
        latent_data = np.load(os.path.join(latent_dir, filename))
        total_samples += len(latent_data)
    
    print(f"Total samples: {total_samples}, Feature dimension: {feature_dim}")
    
    # Create memory-mapped array for efficient storage
    mmap_file = os.path.join(latent_dir, 'latent_representations_mmap.npy')
    if os.path.exists(mmap_file):
        print("Loading existing memory-mapped file...")
        return np.load(mmap_file, mmap_mode='r')
    
    print("Creating memory-mapped file...")
    latent_mmap = np.memmap(mmap_file, dtype='float32', mode='w+', shape=(total_samples, feature_dim))
    
    # Fill the memory-mapped array
    current_idx = 0
    for filename in tqdm(latent_files, desc="Loading Latent Representations"):
        latent_data = np.load(os.path.join(latent_dir, filename))
        batch_size_actual = len(latent_data)
        latent_mmap[current_idx:current_idx + batch_size_actual] = latent_data
        current_idx += batch_size_actual
    
    # Flush to disk
    latent_mmap.flush()
    del latent_mmap
    
    # Return as read-only memory map
    return np.load(mmap_file, mmap_mode='r')

def load_latent_representations_sample(latent_dir, sample_fraction=0.2):
    """Load a sample of latent representations for PCA variance calculation"""
    latent_files = sorted([f for f in os.listdir(latent_dir) if f.startswith('latent_representations_')])
    
    print(f"Found {len(latent_files)} latent representation files")
    
    # Calculate how many files to sample (at least 20% of total)
    n_files_to_sample = max(1, int(len(latent_files) * sample_fraction))
    sampled_files = latent_files[:n_files_to_sample]
    
    print(f"Sampling {len(sampled_files)} out of {len(latent_files)} files ({sample_fraction*100}% of files)")
    
    latent_representations = []
    for filename in tqdm(sampled_files, desc="Loading Sample Latent Representations"):
        latent_data = np.load(os.path.join(latent_dir, filename))
        latent_representations.append(latent_data)
    
    return np.concatenate(latent_representations, axis=0)

def perform_pca_incremental(latent_dir, n_components, batch_size=10000):
    """Perform PCA using incremental learning to handle large datasets"""
    from sklearn.decomposition import IncrementalPCA
    
    latent_files = sorted([f for f in os.listdir(latent_dir) if f.startswith('latent_representations_')])
    
    # Ensure batch_size is at least n_components
    if batch_size < n_components:
        batch_size = n_components + 100  # Add buffer
        print(f"Adjusted batch_size to {batch_size} to ensure it's >= n_components ({n_components})")
    
    # Initialize incremental PCA
    ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
    
    print(f"Fitting incremental PCA on {len(latent_files)} files with {n_components} components...")
    
    # Collect data until we have enough samples for the first batch
    accumulated_data = []
    accumulated_samples = 0
    
    for i, filename in enumerate(tqdm(latent_files, desc="Fitting PCA")):
        latent_data = np.load(os.path.join(latent_dir, filename))
        # Reshape to 2D if needed
        if latent_data.ndim > 2:
            latent_data = latent_data.reshape(latent_data.shape[0], -1)
        
        # Accumulate data until we have enough samples
        accumulated_data.append(latent_data)
        accumulated_samples += len(latent_data)
        
        # Fit when we have enough samples or at the end
        if accumulated_samples >= batch_size or i == len(latent_files) - 1:
            if accumulated_samples >= n_components:  # Ensure we have enough samples
                batch_data = np.concatenate(accumulated_data, axis=0)
                ipca.partial_fit(batch_data)
                print(f"Fitted batch with {len(batch_data)} samples")
            else:
                print(f"Skipping batch with {accumulated_samples} samples (need >= {n_components})")
            
            # Reset accumulation
            accumulated_data = []
            accumulated_samples = 0
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(latent_files)} files")
    
    print("PCA fitting completed. Transforming data...")
    
    # Transform data in batches
    pca_vectors_list = []
    for filename in tqdm(latent_files, desc="Transforming data"):
        latent_data = np.load(os.path.join(latent_dir, filename))
        # Reshape to 2D if needed
        if latent_data.ndim > 2:
            latent_data = latent_data.reshape(latent_data.shape[0], -1)
        pca_batch = ipca.transform(latent_data)
        pca_vectors_list.append(pca_batch)
    
    # Concatenate all transformed vectors
    pca_vectors = np.concatenate(pca_vectors_list, axis=0)
    
    return ipca, pca_vectors

def determine_pca_components_for_variance(data, target_variance=0.99, sample_fraction=0.1):
    """
    Determine the number of PCA components needed to explain target_variance of the variance
    using a subset of the data for efficiency
    """
    # Sample a fraction of the data for variance calculation
    n_samples = int(len(data) * sample_fraction)
    sample_indices = np.random.choice(len(data), n_samples, replace=False)
    data_sample = data[sample_indices]
    
    # Perform PCA on the sample
    pca = PCA()
    pca.fit(data_sample)
    
    # Calculate cumulative explained variance
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    
    # Find the number of components needed
    n_components = np.argmax(cumulative_variance >= target_variance) + 1
    
    print(f"Sample variance explained by {n_components} components: {cumulative_variance[n_components-1]:.4f}")
    print(f"Total variance explained by {n_components} components: {cumulative_variance[n_components-1]:.4f}")
    
    return n_components

def main():
    """Main function for PCA on subset latent space"""
    # Construct paths based on current parameters
    latent_dir = f'{latent_input_folder}{category_folder_name}_latent_seq_{sequence_length}_latent_{latent_dim}_bottleneck_{bottleneck_dim}_no_velocity_test_depth/'
    output_folder = f'{pca_output_folder}{category_folder_name}_pca_seq_{sequence_length}_latent_{latent_dim}_bottleneck_{bottleneck_dim}_no_velocity_test_depth/'

    print(f"Starting PCA analysis for latent space (all categories)...")
    
    # Check if latent directory exists
    if not os.path.exists(latent_dir):
        print(f"Error: Latent directory not found: {latent_dir}")
        print("Please run latent_extraction_relative_features_subset.py first")
        return
    
    # Load frame indices
    print("Loading frame indices...")
    frame_indices = load_frame_indices(latent_dir)
    print(f"Loaded frame indices with shape: {frame_indices.shape}")
    
    # Load a sample of latent representations for variance calculation
    print("Loading sample of latent representations for variance calculation...")
    latent_sample = load_latent_representations_sample(latent_dir, sample_fraction=0.2)
    print(f"Loaded sample latent representations shape: {latent_sample.shape}")
    
    # Reshape the sample data to 2D for PCA
    if latent_sample.ndim > 2:
        latent_sample_2d = latent_sample.reshape(latent_sample.shape[0], -1)
    else:
        latent_sample_2d = latent_sample
    print(f"Reshaped sample latent space shape: {latent_sample_2d.shape}")
    
    # Determine number of PCA components
    print("Determining number of PCA components...")
    target_variance = 0.99
    n_components = determine_pca_components_for_variance(latent_sample_2d, target_variance=target_variance)
    print(f"Determined {n_components} components needed for {target_variance*100}% variance")
    
    # Create output directory
    os.makedirs(output_folder, exist_ok=True)
    
    # Perform incremental PCA on ALL data with determined n_components
    print(f"Performing incremental PCA on all data with {n_components} components...")
    pca, pca_vectors = perform_pca_incremental(latent_dir, n_components)
    print(f"PCA vectors shape: {pca_vectors.shape}")
    
    # Verify that frame indices and PCA vectors have the same number of samples
    if len(frame_indices) != len(pca_vectors):
        print(f"Error: Mismatch between frame indices ({len(frame_indices)}) and PCA vectors ({len(pca_vectors)})")
        return
    
    # Save results
    pca_output_dir = os.path.join(output_folder, 'pca_results')
    os.makedirs(pca_output_dir, exist_ok=True)
    
    # Save PCA vectors
    pca_vectors_file = os.path.join(pca_output_dir, f'pca_vectors_{category_folder_name}_n_components_{n_components}_no_velocity_test_depth_sequence_length_{sequence_length}.npy')
    np.save(pca_vectors_file, pca_vectors)
    print(f"Saved PCA vectors to: {pca_vectors_file}")
    
    # Save PCA model
    pca_model_file = os.path.join(pca_output_dir, f'pca_model_{category_folder_name}_n_components_{n_components}_no_velocity_test_depth_sequence_length_{sequence_length}.pkl')
    with open(pca_model_file, 'wb') as f:
        pickle.dump(pca, f)
    print(f"Saved PCA model to: {pca_model_file}")
    
    # Save frame indices for reference
    frame_indices_file = os.path.join(pca_output_dir, f'frame_indices_{category_folder_name}_no_velocity_test_depth_sequence_length_{sequence_length}.npy')
    np.save(frame_indices_file, frame_indices)
    print(f"Saved frame indices to: {frame_indices_file}")
    
    # Load and save frame indices mapping if it exists
    mapping_file = os.path.join(latent_dir, 'frame_indices_mapping.pkl')
    if os.path.exists(mapping_file):
        with open(mapping_file, 'rb') as f:
            frame_indices_mapping = pickle.load(f)
        
        mapping_save_file = os.path.join(pca_output_dir, f'frame_indices_mapping_{category_folder_name}_no_velocity_test_depth_sequence_length_{sequence_length}.pkl')
        with open(mapping_save_file, 'wb') as f:
            pickle.dump(frame_indices_mapping, f)
        print(f"Saved frame indices mapping to: {mapping_save_file}")
    
    # Print summary
    print(f"\n=== PCA SUMMARY ===")
    print(f"Total frames: {len(frame_indices)}")
    print(f"PCA vectors shape: {pca_vectors.shape}")
    print(f"Sample latent shape (for variance): {latent_sample_2d.shape}")
    print(f"PCA components: {n_components}")
    print(f"Target variance explained: {target_variance*100}%")
    print(f"Actual variance explained: {np.sum(pca.explained_variance_ratio_)*100:.2f}%")
    print(f"Results saved to: {pca_output_dir}")
    
    print(f"\nPCA analysis completed successfully!")

if __name__ == "__main__":
    main() 