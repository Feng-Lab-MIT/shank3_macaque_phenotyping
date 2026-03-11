import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import pickle

# Load data
all_dfs = pd.read_pickle("/media/fenglab/Seagate Hub/Shank3_macaque/second cohort/social/limb_model/all_dfs_with_limb_features.pkl")

# Load category file for filtering
category_file = '/media/fenglab/New Volume/River_data_backup/2nd_cohort_social/3d/model/data_for_analysis_200_1_5_64_clusters.csv'
filtered_df = pd.read_csv(category_file)

# Set random seed FIRST to ensure reproducibility (same as training)
random_seed = 42
np.random.seed(random_seed)

# Model parameters - change these to match your trained model
sequence_length = 240
latent_dim = 64
bottleneck_dim = 32
prediction_length = 24

# Other Parameters
batch_size = 32
original_sequence_length = 480  # The sequence length used when selecting category data
# Target categories for filtering
desired_categories = ['Bottom Playing', 'Bottom Alone']
include_depth_features = True  # Whether to include absolute depth features

# Folder paths - can be updated by pipeline
model_folder = '/home/fenglab/Documents/dlc/attention_model_scripts/model_2_sub_train_with_sub/'
latent_output_folder = '/media/fenglab/newssd/social/'

def get_loc_feature_indices(column_name_after_der_dict):
    """Get indices of location related features - exact same as model training"""
    loc_feature_indices = []
    for i in column_name_after_der_dict:
        s = column_name_after_der_dict[i]
        substrings = ['_x','_y','_z','_x_ma','_y_ma','_z_ma','_x_ma_der','_y_ma_der','_z_ma_der']
        if 'depth' in s:
            loc_feature_indices.append(i)
        elif any(s.endswith(sub) for sub in substrings):
            loc_feature_indices.append(i)
    return loc_feature_indices

def get_additional_column_indices(column_name_after_der_dict):
    """Get indices of features to exclude from output (features we don't want to predict)"""
    additional_feature_indices = []
    for i in column_name_after_der_dict:
        s = column_name_after_der_dict[i]
        # Exclude ALL probe features from output (regardless of type)
        if s.startswith('probe') or s.startswith('log_probe') or 'probe_' in s:
            additional_feature_indices.append(i)
        # Exclude x,y,z features
        elif '_x_ma' in s or '_y_ma' in s or '_z_ma' in s:
            additional_feature_indices.append(i)
        # Exclude non-angle features that are not ma or der
        elif 'angle' not in s:
            if not s.endswith('ma') and not s.endswith('ma1') and not s.endswith('der'):
                additional_feature_indices.append(i)
    return additional_feature_indices

def compute_derivatives_and_return_column_names(data, column_names, exclude_columns=[]):
    """Compute derivatives of features"""
    derivatives = np.diff(data, axis=0, prepend=data[0:1, :])
    included_indices = [i for i in range(data.shape[1]) if i not in exclude_columns]
    derivatives = derivatives[:, included_indices]
    derivative_column_names = [f'{column_names[i]}_der' for i in included_indices]
    return derivatives, derivative_column_names

def compute_moving_average_and_return_column_names(data, column_names, window=24, exclude_columns=[]):
    """Compute moving averages of features"""
    data_df = pd.DataFrame(data, columns=column_names)
    moving_averages = data_df.rolling(window=window, min_periods=1).mean().to_numpy()
    included_indices = [i for i in range(data.shape[1]) if i not in exclude_columns]
    moving_averages = moving_averages[:, included_indices]
    moving_average_column_names = [f'{column_names[i]}_ma' for i in included_indices]
    return moving_averages, moving_average_column_names

def convert_to_relative_features(data, column_names):
    """
    Convert location features to relative features (test - probe)
    Handles special cases where test features don't have 'test_' prefix
    Only keeps relative features, removes all original location features
    """
    
    # Get location feature indices
    column_name_dict = {i: name for i, name in enumerate(column_names)}
    loc_feature_indices = get_loc_feature_indices(column_name_dict)
    
    # Dictionary to map test features to their probe counterparts
    test_to_probe_mapping = {}
    
    # Find test-probe pairs for hand and foot features
    test_probe_pairs = {
        'left_hand_x': 'probe_left_hand_x',
        'left_hand_y': 'probe_left_hand_y', 
        'left_hand_z': 'probe_left_hand_z',
        'right_hand_x': 'probe_right_hand_x',
        'right_hand_y': 'probe_right_hand_y',
        'right_hand_z': 'probe_right_hand_z',
        'left_foot_x': 'probe_left_foot_x',
        'left_foot_y': 'probe_left_foot_y',
        'left_foot_z': 'probe_left_foot_z',
        'right_foot_x': 'probe_right_foot_x',
        'right_foot_y': 'probe_right_foot_y',
        'right_foot_z': 'probe_right_foot_z'
    }
    
    # Handle special cases where test features don't have 'test_' prefix
    special_cases = {
        'depth': 'probe_depth',
        'spinal_cord_depth': 'probe_spinal_cord_depth'
    }
    
    # Find all test-probe pairs
    for test_name, probe_name in test_probe_pairs.items():
        if test_name in column_names and probe_name in column_names:
            test_idx = column_names.index(test_name)
            probe_idx = column_names.index(probe_name)
            test_to_probe_mapping[test_idx] = probe_idx
    
    for test_name, probe_name in special_cases.items():
        if test_name in column_names and probe_name in column_names:
            test_idx = column_names.index(test_name)
            probe_idx = column_names.index(probe_name)
            test_to_probe_mapping[test_idx] = probe_idx
    
    # Convert to relative features
    relative_features = []
    relative_feature_names = []
    
    for test_idx, probe_idx in test_to_probe_mapping.items():
        if test_idx in loc_feature_indices and probe_idx in loc_feature_indices:
            # Calculate relative feature: test - probe
            relative_feature = data[:, test_idx] - data[:, probe_idx]
            relative_features.append(relative_feature)
            
            # Create relative feature name
            test_name = column_names[test_idx]
            if test_name.startswith('test_'):
                relative_name = test_name.replace('test_', 'relative_')
            else:
                # For features without 'test_' prefix, add 'relative_' prefix
                relative_name = f'relative_{test_name}'
            
            relative_feature_names.append(relative_name)
    
    # Create new data with only relative features and non-location features
    if relative_features:
        relative_features = np.column_stack(relative_features)
        
        # Remove ALL original location features (both test and probe)
        all_location_features_to_remove = []
        for i, name in enumerate(column_names):
            if i in loc_feature_indices:
                all_location_features_to_remove.append(i)
        
        # Keep only non-location features
        features_to_keep = [i for i in range(data.shape[1]) if i not in all_location_features_to_remove]
        
        data_relative = data[:, features_to_keep]
        new_column_names = [column_names[i] for i in features_to_keep]
        
        # Add relative features
        data_relative = np.column_stack([data_relative, relative_features])
        new_column_names.extend(relative_feature_names)
        
        return data_relative, new_column_names
    else:
        return data, column_names

def preprocess_data_with_relative_features_subset(all_dfs, sequence_length, prediction_length, filtered_df, original_df_indices):
    """
    Preprocess data with relative location features for subset of target frames
    """
    print("Starting data preprocessing with relative features for subset...")
    
    # Print all feature names from the first dataframe for debugging
    if all_dfs:
        first_df = all_dfs[0]
        print("\n=== ORIGINAL FEATURE NAMES (before preprocessing) ===")
        for i, col in enumerate(first_df.columns):
            if col != 'Monkey_Probe':
                print(f"{i:3d}: {col}")
        print(f"Total features (excluding Monkey_Probe): {len(first_df.columns) - 1}")
        print("=" * 60)
    
    all_data_combined = []
    all_column_names = []
    index_tracker = []
    frame_indices_mapping = []  # Track which frames are kept
    
    # Generate di_mp_dic mapping (same as in latent extraction script)
    pre_sample_feature = pd.read_csv('/media/fenglab/New Volume/River_data_backup/2nd_cohort_social/3d/features/pre/probe_z_calib_0511.csv', index_col=[0], header=[0])
    pre_monkey_day_list = [col for col in pre_sample_feature.columns if '_' in col]
    post_sample_feature = pd.read_csv('/media/fenglab/New Volume/River_data_backup/2nd_cohort_social/3d/features/post/probe_z_calib_202309.csv', index_col=[0], header=[0])
    post_monkey_day_list = [col for col in post_sample_feature.columns if '_' in col]
    
    monkey_day_list = pre_monkey_day_list + post_monkey_day_list
    monkey_day_list.sort()
    
    # Create the mapping dictionary
    di_mp_dic = {i: s for i, s in enumerate(monkey_day_list)}
    print(f"Generated di_mp_dic with {len(di_mp_dic)} entries")
    
    # Create a simple set of (DataFrame Index, Frame Index) tuples for quick lookup
    target_frames = set()
    for idx, row in filtered_df.iterrows():
        df_index = row['DataFrame Index']
        frame_index = row['Frame Index']
        target_frames.add((df_index, frame_index))
    
    print(f"Created set of {len(target_frames)} target frame starting points")
    
    for local_index, df in enumerate(all_dfs):
        # Get the original dataframe index
        original_df_index = original_df_indices[local_index]
        
        # Remove Monkey_Probe column
        data = df.loc[:, df.columns != 'Monkey_Probe'].to_numpy()
        
        # Get column names
        column_names = [col for col in df.columns if col != 'Monkey_Probe']
        
        # Check if this dataframe has any target frames
        dataframe_target_frames = [frame_idx for df_idx, frame_idx in target_frames if df_idx == original_df_index]
        
        if not dataframe_target_frames:
            print(f"No target frames found for dataframe {original_df_index}")
            continue
        
        print(f"Dataframe {original_df_index}: processing all {len(data)} frames, will use {len(dataframe_target_frames)} as sequence starting points")
        
        # Impute missing values
        imputer = SimpleImputer(strategy='mean')
        data = imputer.fit_transform(data)
        
        # Convert to relative features
        data, column_names = convert_to_relative_features(data, column_names)
        
        # Optionally add absolute depth features (test_depth and probe_depth)
        if include_depth_features:
            # Extract depth features from original data before relative conversion
            depth_features = []
            depth_feature_names = []
            
            # Find depth and probe_depth columns in original data
            for col in df.columns:
                if col != 'Monkey_Probe' and ('depth' in col or 'probe_depth' in col):
                    depth_values = df[col].values
                    # Check for NaN values and handle them
                    nan_count = np.sum(np.isnan(depth_values))
                    if nan_count > 0:
                        print(f"Warning: Found {nan_count} NaN values in {col} (out of {len(depth_values)} total), filling with mean")
                        depth_values = np.nan_to_num(depth_values, nan=np.nanmean(depth_values))
                    else:
                        print(f"No NaN values found in {col}")
                    depth_features.append(depth_values)
                    depth_feature_names.append(col)
            
            if depth_features:
                # Stack depth features
                depth_data = np.column_stack(depth_features)
                # Check for any remaining NaN values
                if np.any(np.isnan(depth_data)):
                    print("Warning: Found NaN values in depth data, filling with 0")
                    depth_data = np.nan_to_num(depth_data, nan=0.0)
                # Add to the relative features
                data = np.concatenate((data, depth_data), axis=1)
                column_names.extend(depth_feature_names)
                print(f"Added {len(depth_feature_names)} depth features: {depth_feature_names}")
                print(f"Depth data shape: {depth_data.shape}, contains NaN: {np.any(np.isnan(depth_data))}")
        else:
            print("Skipping depth features (include_depth_features=False)")
        
        # Compute moving averages
        ma_excluded_columns = [i for i, name in enumerate(column_names) if 'ma1' in name or 'angle' in name]
        mas, ma_column_names = compute_moving_average_and_return_column_names(
            data, column_names, exclude_columns=ma_excluded_columns
        )
        
        # Add moving averages
        data = np.concatenate((data, mas), axis=1)
        column_names.extend(ma_column_names)
        
        # Compute derivatives
        der_excluded_columns = [i for i, name in enumerate(column_names) 
                              if not name.endswith('ma') and not name.endswith('ma1')]
        derivatives, der_column_names = compute_derivatives_and_return_column_names(
            data, column_names, exclude_columns=der_excluded_columns
        )
        
        # Add derivatives
        data = np.concatenate((data, derivatives), axis=1)
        column_names.extend(der_column_names)
        
        all_data_combined.append(data)
        all_column_names.append(column_names)
        index_tracker.append((original_df_index, len(data)))
        
        # Store the mapping from processed frame index to original frame index
        for frame_idx in dataframe_target_frames:
            frame_indices_mapping.append((original_df_index, frame_idx, local_index))
    
    # Concatenate all data
    all_data = np.concatenate(all_data_combined, axis=0)
    final_column_names = all_column_names[0]  # Use first dataframe's column names as reference
    
    print(f"Final data shape before velocity filtering: {all_data.shape}")
    print(f"Number of input features before velocity filtering: {all_data.shape[1]}")
    
    # Filter out velocity-related features
    print("\n=== FILTERING OUT VELOCITY FEATURES ===")
    velocity_feature_indices = []
    for i, feature_name in enumerate(final_column_names):
        if 'velocity' in feature_name.lower():
            velocity_feature_indices.append(i)
    
    if velocity_feature_indices:
        print(f"Found {len(velocity_feature_indices)} velocity-related features to exclude:")
        for idx in velocity_feature_indices:
            print(f"  {idx}: {final_column_names[idx]}")
        
        # Create mask to keep non-velocity features
        keep_indices = [i for i in range(len(final_column_names)) if i not in velocity_feature_indices]
        
        # Filter data and column names
        all_data = all_data[:, keep_indices]
        final_column_names = [final_column_names[i] for i in keep_indices]
        
        print(f"Excluded {len(velocity_feature_indices)} velocity features")
        print(f"Final data shape after velocity filtering: {all_data.shape}")
        print(f"Number of input features after velocity filtering: {all_data.shape[1]}")
    else:
        print("No velocity-related features found to exclude")
    
    # Check for NaN values in final data
    if np.any(np.isnan(all_data)):
        print("Warning: Found NaN values in final data, filling with 0")
        all_data = np.nan_to_num(all_data, nan=0.0)
        print("NaN values replaced with 0")
    else:
        print("No NaN values found in final data")
    
    print(f"Total frames kept: {len(frame_indices_mapping)}")
    
    # Standardize the data (crucial for neural network training)
    print("\n=== STANDARDIZING DATA ===")
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    all_data = scaler.fit_transform(all_data)
    print("Data standardized using StandardScaler (mean=0, std=1)")
    print(f"Data statistics after standardization:")
    print(f"  Mean: {np.mean(all_data):.6f}")
    print(f"  Std: {np.std(all_data):.6f}")
    print(f"  Min: {np.min(all_data):.6f}")
    print(f"  Max: {np.max(all_data):.6f}")
    
    # Create standardized data for each dataframe to maintain proper indexing
    all_data_standardized = []
    current_idx = 0
    for i, data in enumerate(all_data_combined):
        data_len = len(data)
        standardized_data = all_data[current_idx:current_idx + data_len]
        all_data_standardized.append(standardized_data)
        current_idx += data_len
        print(f"Dataframe {i}: {data_len} frames, standardized shape: {standardized_data.shape}")
    
    print(f"Created {len(all_data_standardized)} standardized dataframes")
    
    return all_data, final_column_names, index_tracker, frame_indices_mapping, scaler, all_data_standardized

def generate_sequences_for_latent_extraction_subset(all_data_standardized, index_tracker, frame_indices_mapping, sequence_length, prediction_length, original_sequence_length):
    """
    Generate sequences for latent extraction following the same pattern as training
    But only for frames that match the category filter
    Adjusts starting positions based on sequence length difference
    """
    print("Generating sequences for latent extraction from subset...")
    print(f"Current sequence length: {sequence_length}")
    print(f"Original sequence length (used for category selection): {original_sequence_length}")
    
    # Debug: Check the shape of the first dataframe
    if len(all_data_standardized) > 0:
        print(f"Debug: First dataframe shape: {all_data_standardized[0].shape}")
    
    # Calculate the offset needed to adjust starting positions
    sequence_offset = original_sequence_length - sequence_length
    print(f"Sequence offset adjustment: {sequence_offset} frames")
    
    all_sequences = []
    all_indices = []
    
    for i, (df_index, length) in enumerate(index_tracker):
        data = all_data_standardized[i]  # Use standardized dataframe directly
        print(f"Debug: data shape for dataframe {i}: {data.shape}")
        
        # Get target frames for this dataframe
        dataframe_target_frames = [frame_idx for df_idx, frame_idx, local_idx in frame_indices_mapping if df_idx == df_index]
        
        # Generate sequences only from target frames
        for frame_index in dataframe_target_frames:
            # Adjust the starting position based on sequence length difference
            adjusted_start_index = frame_index + sequence_offset
            
            # Check if we have enough frames for the sequence
            if adjusted_start_index >= 0 and adjusted_start_index + sequence_length + prediction_length <= len(data):
                X_seq = data[adjusted_start_index:adjusted_start_index + sequence_length]
                decoder_input = data[adjusted_start_index + sequence_length - 1:adjusted_start_index + sequence_length - 1 + prediction_length]
                
                # Debug print for first sequence
                if len(all_sequences) == 0:
                    print(f"Debug: data shape: {data.shape}")
                    print(f"Debug: X_seq shape: {X_seq.shape}")
                    print(f"Debug: decoder_input shape: {decoder_input.shape}")
                    print(f"Debug: adjusted_start_index: {adjusted_start_index}, sequence_length: {sequence_length}")
                
                all_sequences.append((X_seq, decoder_input))
                # Store the original frame index (not the adjusted one) for reference
                all_indices.append((df_index, frame_index))
            else:
                print(f"Warning: Not enough frames for sequence starting at adjusted index {adjusted_start_index} in dataframe {df_index}")
    
    print(f"Generated {len(all_sequences)} sequences from {len(frame_indices_mapping)} target frames")
    return all_sequences, all_indices

def extract_latent_space(latent_space_model, sequences, indices, output_dir, batch_size=32):
    """Extract latent space representations"""
    print(f"\n=== EXTRACTING LATENT SPACE ===")
    print(f"Number of sequences: {len(sequences)}")
    print(f"Batch size: {batch_size}")
    print(f"Output directory: {output_dir}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Suppress TensorFlow logging
    tf.get_logger().setLevel('ERROR')
    
    latent_representations = []
    all_indices = []
    batch_num = 0
    
    # Process in batches
    for i in tqdm(range(0, len(sequences), batch_size), desc="Extracting latent space"):
        batch_sequences = sequences[i:i + batch_size]
        batch_indices = indices[i:i + batch_size]
        
        # Prepare batch data
        X_batch = np.array([seq[0] for seq in batch_sequences])
        decoder_input_batch = np.array([seq[1] for seq in batch_sequences])
        
        # Debug print for first batch
        if i == 0:
            print(f"Debug: batch_sequences length: {len(batch_sequences)}")
            print(f"Debug: first sequence shape: {batch_sequences[0][0].shape}")
            print(f"Debug: first sequence type: {type(batch_sequences[0][0])}")
            print(f"Debug: first sequence content: {batch_sequences[0][0]}")
            print(f"Debug: X_batch shape: {X_batch.shape}")
            print(f"Debug: decoder_input_batch shape: {decoder_input_batch.shape}")
        
        # Extract latent representations
        latent_batch = latent_space_model.predict([X_batch, decoder_input_batch], verbose=0)
        
        # Store results
        latent_representations.extend(latent_batch)
        all_indices.extend(batch_indices)
        
        # Save in chunks
        if len(latent_representations) >= 1000:  # Save every 1000 samples
            np.save(os.path.join(output_dir, f'latent_representations_{batch_num}.npy'), np.array(latent_representations))
            np.save(os.path.join(output_dir, f'indices_{batch_num}.npy'), np.array(all_indices))
            
            latent_representations = []
            all_indices = []
            batch_num += 1
    
    # Save remaining data
    if latent_representations:
        np.save(os.path.join(output_dir, f'latent_representations_{batch_num}.npy'), np.array(latent_representations))
        np.save(os.path.join(output_dir, f'indices_{batch_num}.npy'), np.array(all_indices))
    
    # Reset TensorFlow logging
    tf.get_logger().setLevel('INFO')

def main():
    """Main function for latent extraction"""
    print("Starting latent extraction for relative feature model (clean version)...")
    
    # Configure GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if len(gpus) > 1:
        print(f"Using GPU 1: {gpus[1]}")
        tf.config.experimental.set_visible_devices(gpus[1], 'GPU')
    elif len(gpus) == 1:
        print(f"Using GPU 0: {gpus[0]}")
    else:
        print("No GPU found, using CPU")
    
    # Load the trained model (updated to use standardized model)
    model_path = f'{model_folder}best_model_relative_features_with_depth_Bottom_Playing_and_Alone_decoder_fixed_v4_enhanced_standardized_no_velocity_test_depth_output_seq{sequence_length}_latent{latent_dim}_bottleneck{bottleneck_dim}.h5'
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return
    
    print(f"Loading model from {model_path}")
    model = load_model(model_path)
    
    # Print model layers to identify bottleneck layer
    print("\nModel layers:")
    for layer in model.layers:
        print(f"{layer.name}: {layer.output_shape}")
    
    # Load data
    print("Loading data...")
    all_dfs = pd.read_pickle("/media/fenglab/Seagate Hub/Shank3_macaque/second cohort/social/limb_model/all_dfs_with_limb_features.pkl")
    
    # Load the behavioral category data
    category_file = '/media/fenglab/New Volume/River_data_backup/2nd_cohort_social/3d/model/data_for_analysis_200_1_5_64_clusters.csv'
    print(f"Loading behavioral category data from: {category_file}")
    category_df = pd.read_csv(category_file)
    print(f"Loaded category data with shape: {category_df.shape}")
    
    # Filter for desired categories
    filtered_df = category_df[category_df['Category'].isin(desired_categories)]
    print(f"Filtered to {len(filtered_df)} frames with categories: {desired_categories}")
    
    # Use all dataframes (same as training script)
    original_df_indices = np.arange(len(all_dfs))
    
    # Preprocess data using subset function
    print(f"\n=== PREPROCESSING DATA ===")
    print(f"Number of dataframes: {len(all_dfs)}")
    print(f"Filtered dataframe shape: {filtered_df.shape}")
    print(f"Desired categories: {desired_categories}")
    
    all_data, final_column_names, index_tracker, frame_indices_mapping, scaler, all_data_standardized = preprocess_data_with_relative_features_subset(
        all_dfs, sequence_length, prediction_length, filtered_df, original_df_indices
    )
    
    print(f"Preprocessing completed!")
    print(f"Final data shape: {all_data.shape}")
    print(f"Number of standardized dataframes: {len(all_data_standardized)}")
    print(f"Number of features: {len(final_column_names)}")
    
    # Generate sequences
    print(f"\n=== GENERATING SEQUENCES ===")
    total_frames = sum(len(df) for df in all_data_standardized)
    print(f"Total frames across all dataframes: {total_frames}")
    print(f"Sequence length: {sequence_length}")
    print(f"Prediction length: {prediction_length}")
    print(f"Original sequence length: {original_sequence_length}")
    print(f"Total possible sequences: {total_frames - sequence_length + 1}")
    
    sequences, indices = generate_sequences_for_latent_extraction_subset(
        all_data_standardized, index_tracker, frame_indices_mapping, sequence_length, prediction_length, original_sequence_length
    )
    
    print(f"Generated {len(sequences)} sequences total")
    print(f"Debug: Type of sequences: {type(sequences)}")
    print(f"Debug: Length of sequences: {len(sequences)}")
    if len(sequences) > 0:
        print(f"Debug: Type of first element: {type(sequences[0])}")
        if isinstance(sequences[0], tuple) and len(sequences[0]) > 0:
            print(f"Debug: First element is a tuple with {len(sequences[0])} elements")
            print(f"Debug: Type of first element[0]: {type(sequences[0][0])}")
    
    # Define the latent space model (using bottleneck layer)
    bottleneck_layer_name = 'dense_1'  # This should be the second dense layer (bottleneck2)
    try:
        latent_space_model = Model(
            inputs=[model.input[0], model.input[1]], 
            outputs=model.get_layer(bottleneck_layer_name).output
        )
        print(f"Using bottleneck layer: {bottleneck_layer_name}")
        print(f"Bottleneck layer output shape: {model.get_layer(bottleneck_layer_name).output_shape}")
    except:
        # If specific layer not found, try to find the second dense layer
        dense_layers = [layer for layer in model.layers if 'dense' in layer.name and layer != model.layers[-1]]
        if len(dense_layers) >= 2:
            # Use the second-to-last dense layer (should be bottleneck2)
            bottleneck_layer_name = dense_layers[-2].name
            latent_space_model = Model(
                inputs=[model.input[0], model.input[1]], 
                outputs=dense_layers[-2].output
            )
            print(f"Using second dense layer: {bottleneck_layer_name}")
            print(f"Bottleneck layer output shape: {dense_layers[-2].output_shape}")
        else:
            print("Error: Could not find appropriate bottleneck layer")
            return
    
    # Extract latent space
    print(f"\n=== EXTRACTING LATENT SPACE ===")
    print(f"Number of sequences: {len(sequences[0])}")
    print(f"Sequence shape: {sequences[0][0].shape if len(sequences[0]) > 0 else 'No sequences'}")
    print(f"Batch size: {batch_size}")
    
    output_dir = f"{latent_output_folder}bottom_playing_and_alone_latent_seq_{sequence_length}_latent_{latent_dim}_bottleneck_{bottleneck_dim}_no_velocity_test_depth/"
    print(f"Output directory: {output_dir}")
    
    extract_latent_space(latent_space_model, sequences, indices, output_dir, batch_size)
    
    print(f"Latent extraction completed. Results saved to {output_dir}")

if __name__ == "__main__":
    main()