#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model training script with relative location features
Converts location-related features to relative features (test - probe)
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import itertools
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, CSVLogger
from tensorflow.keras.optimizers.schedules import CosineDecay
from tensorflow.keras.layers import Input, LSTM, TimeDistributed, Dense, MultiHeadAttention, Concatenate, LayerNormalization, Dropout, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Activation
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.utils import disable_interactive_logging
from sklearn.metrics import mean_squared_error
from tensorflow.keras.backend import clear_session
import tensorflow
from tqdm import tqdm
import glob
import cv2
import joblib
import pickle

# Clear Keras session
K.clear_session()

# GPU configuration - Use GPU 1
gpus = tf.config.experimental.list_physical_devices('GPU')
if len(gpus) > 1:
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    print(f"Using GPU 0: {gpus[0]}")
elif len(gpus) == 1:
    print(f"Only one GPU available, using GPU 0: {gpus[0]}")
else:
    print("No GPU found, using CPU")

# Configure GPU memory growth
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Global parameters
folder = '/home/fenglab/Documents/dlc/attention_model_scripts/model_2_sub_train_with_sub/'

# Model parameters
sequence_lengths = [240]  # 5,10,20,40,60s
prediction_lengths = [24]  # 1,2s
original_sequence_length = 480  # The sequence length used when selecting category data
latent_dims = [64]  # Increased latent dimension for LSTM (was 64)
test_size = 0.3  # Proportion of data for validation
dropout_rates = [0.2]  # Original dropout rate
regularization_lambdas = [0.1]  # Original L2 regularization
bottleneck_dims = [32]  # Increased bottleneck dimension (was 32)
patience = 30
batch_size = 32
include_depth_features = True  # Whether to include absolute depth features

# Set random seed for reproducibility
random_seed = 42
np.random.seed(random_seed)

def get_loc_feature_indices(column_name_after_der_dict):
    """Get indices of location related features"""
    loc_feature_indices = []
    for i in column_name_after_der_dict:
        s = column_name_after_der_dict[i]
        substrings = ['_x','_y','_z','_x_ma','_y_ma','_z_ma','_x_ma_der','_y_ma_der','_z_ma_der']
        if 'depth' in s:
            loc_feature_indices.append(i)
        elif any(s.endswith(sub) for sub in substrings):
            loc_feature_indices.append(i)
    return loc_feature_indices

def get_post_feature_indices(column_name_after_der_dict):
    """Get indices of posture related features"""
    post_feature_indices = []
    for i in column_name_after_der_dict:
        s = column_name_after_der_dict[i]
        if 'angle' in s:
            post_feature_indices.append(i)
    return post_feature_indices

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

def create_sequences(data, sequence_length, prediction_step=48):
    """Create sequences for training"""
    X, y = [], []
    for i in range(len(data) - sequence_length - prediction_step):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length + prediction_step - 1])
    return np.array(X), np.array(y)

def temporal_within_dataframe_split(all_data_combined, index_tracker, sequence_length, test_size=0.3):
    """
    Split frames temporally within each dataframe (first 70% train, last 30% val)
    This ensures all subjects are represented while minimizing temporal overlap
    Each frame index represents a starting point for sequence generation
    """
    train_data_combined = []
    val_data_combined = []
    
    for i, (df_index, length) in enumerate(index_tracker):
        # Get the data for this dataframe (already a numpy array)
        data = all_data_combined[i]
        
        # Calculate split point (70% for training, 30% for validation)
        num_frames = len(data)
        num_train_frames = int(num_frames * (1 - test_size))
        
        # Split the data temporally
        train_frames = data[:num_train_frames]
        val_frames = data[num_train_frames:]
        
        train_data_combined.append(train_frames)
        val_data_combined.append(val_frames)
        
        print(f"Dataframe {i} (index {df_index}): {num_frames} total frames -> {len(train_frames)} train, {len(val_frames)} val")
    
    # Concatenate all frames
    train_data = np.concatenate(train_data_combined, axis=0)
    val_data = np.concatenate(val_data_combined, axis=0)
    
    print(f"Total training data shape: {train_data.shape}")
    print(f"Total validation data shape: {val_data.shape}")
    print(f"Training/validation ratio: {len(train_data)/(len(train_data)+len(val_data)):.3f}")
    
    return train_data, val_data

def random_frame_train_test_split(all_data_combined, index_tracker, sequence_length, test_size=0.3, random_seed=42):
    """
    Split frames randomly within each dataframe for training and validation
    This ensures all subjects are represented in both training and validation sets
    DEPRECATED: Use temporal_within_dataframe_split to minimize temporal overlap
    """
    np.random.seed(random_seed)
    
    train_data_combined = []
    val_data_combined = []
    
    for i, (df_index, length) in enumerate(index_tracker):
        # Get the data for this dataframe (already a numpy array)
        data = all_data_combined[i]
        
        # Randomly sample frames from this dataframe
        num_frames = len(data)
        num_test_frames = int(num_frames * test_size)
        
        # Create random indices for test set
        test_indices = np.random.choice(num_frames, num_test_frames, replace=False)
        train_indices = [idx for idx in range(num_frames) if idx not in test_indices]
        
        # Split the data
        train_frames = data[train_indices]
        val_frames = data[test_indices]
        
        train_data_combined.append(train_frames)
        val_data_combined.append(val_frames)
        
        print(f"Dataframe {i} (index {df_index}): {num_frames} total frames -> {len(train_indices)} train, {len(test_indices)} val")
    
    # Concatenate all frames
    train_data = np.concatenate(train_data_combined, axis=0)
    val_data = np.concatenate(val_data_combined, axis=0)
    
    print(f"Total training data shape: {train_data.shape}")
    print(f"Total validation data shape: {val_data.shape}")
    print(f"Training/validation ratio: {len(train_data)/(len(train_data)+len(val_data)):.3f}")
    
    return train_data, val_data

def temporal_train_test_split(data, sequence_length, test_size=0.3):
    """Split data temporally (kept for backward compatibility)"""
    num_train = int((1 - test_size) * len(data))
    train_data = data[:num_train]
    val_data = data[num_train - sequence_length:]
    return train_data, val_data

def data_generator_with_additional_features(data, sequence_length, prediction_length, batch_size, random_seed, additional_feature_indices):
    """Data generator for encoder-decoder model that excludes additional features from output"""
    rng = np.random.default_rng(random_seed)
    while True:
        encoder_batch, decoder_batch, y_batch = [], [], []
        while len(encoder_batch) < batch_size:
            start_idx = rng.integers(0, len(data) - sequence_length - prediction_length)
            
            # Encoder input: sequence_length frames
            encoder_seq = data[start_idx:start_idx + sequence_length]
            
            # Decoder input: zeros (model learns to generate from encoder context)
            decoder_seq = np.zeros((prediction_length, data.shape[1]))
            
            # Target: prediction_length frames (exclude additional features)
            target_seq = data[start_idx + sequence_length:start_idx + sequence_length + prediction_length]
            target_seq_excluded = np.delete(target_seq, additional_feature_indices, axis=1)
            
            encoder_batch.append(encoder_seq)
            decoder_batch.append(decoder_seq)
            y_batch.append(target_seq_excluded)
        
        encoder_batch = np.array(encoder_batch)
        decoder_batch = np.array(decoder_batch)
        y_batch = np.array(y_batch)
        
        yield [encoder_batch, decoder_batch], y_batch

def data_generator_from_sequences(sequences, sequence_length, prediction_length, batch_size, random_seed, additional_feature_indices):
    """Data generator for encoder-decoder model that works with pre-generated sequences"""
    rng = np.random.default_rng(random_seed)
    available_indices = []
    current_idx = 0
    
    while True:
        encoder_batch, decoder_batch, y_batch = [], [], []
        while len(encoder_batch) < batch_size:
            # If we've used all sequences, shuffle for new epoch
            if current_idx >= len(available_indices):
                available_indices = np.arange(len(sequences))
                rng.shuffle(available_indices)
                current_idx = 0
            
            # Get next sequence index
            seq_idx = available_indices[current_idx]
            current_idx += 1
            input_seq, target_seq = sequences[seq_idx]
            
            # Encoder input: sequence_length frames
            encoder_seq = input_seq
            
            # Decoder input: prediction_length frames (same as backup)
            decoder_seq = target_seq
            
            # Target: prediction_length frames (exclude additional features)
            target_seq_excluded = np.delete(target_seq, additional_feature_indices, axis=1)
            
            encoder_batch.append(encoder_seq)
            decoder_batch.append(decoder_seq)
            y_batch.append(target_seq_excluded)
        
        encoder_batch = np.array(encoder_batch)
        decoder_batch = np.array(decoder_batch)
        y_batch = np.array(y_batch)
        
        
        yield [encoder_batch, decoder_batch], y_batch

def learning_rate_schedule(epoch, initial_lr=1e-4, decay_factor=0.5, patience=10):
    """
    Simple learning rate scheduler - just use constant learning rate
    """
    # Keep learning rate constant for now
    lr = initial_lr
    
    print(f"Epoch {epoch}: Learning rate = {lr:.2e}")
    return lr

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
    
    # Plot feature distributions after standardization (optional)
    from scipy import stats  # Import at function level to avoid scope issues
    
    feature_plots_folder = f'{folder}feature_distributions/'
    
    if os.path.exists(feature_plots_folder):
        print(f"\n=== SKIPPING FEATURE DISTRIBUTION PLOTTING ===")
        print(f"Feature distributions folder already exists: {feature_plots_folder}")
        print("Skipping plotting to save time. Delete the folder if you want to regenerate plots.")
    else:
        print("\n=== PLOTTING FEATURE DISTRIBUTIONS ===")
        
        # Create folder for individual feature plots
        os.makedirs(feature_plots_folder, exist_ok=True)
        
        n_features = all_data.shape[1]
        
        # Plot each feature individually
        for i in range(n_features):
            # Create individual figure for each feature
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            
            # Plot histogram
            ax.hist(all_data[:, i], bins=50, alpha=0.7, density=True, color='skyblue', edgecolor='black')
            
            # Add normal distribution overlay
            x = np.linspace(all_data[:, i].min(), all_data[:, i].max(), 100)
            normal_dist = stats.norm.pdf(x, 0, 1)  # Standard normal
            ax.plot(x, normal_dist, 'r-', linewidth=2, label='Standard Normal')
            
            # Calculate statistics
            mean_val = np.mean(all_data[:, i])
            std_val = np.std(all_data[:, i])
            skewness = stats.skew(all_data[:, i])
            kurtosis = stats.kurtosis(all_data[:, i])
            
            # Set title and labels
            feature_name = final_column_names[i]
            ax.set_title(f'{feature_name}\nμ={mean_val:.3f}, σ={std_val:.3f}, skew={skewness:.3f}, kurt={kurtosis:.3f}', 
                        fontsize=12, pad=15)
            ax.set_xlabel('Value')
            ax.set_ylabel('Density')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # Save individual plot
            safe_feature_name = feature_name.replace('/', '_').replace('\\', '_').replace(':', '_')
            feature_plot_filename = f'{feature_plots_folder}feature_{i:03d}_{safe_feature_name}.png'
            plt.tight_layout()
            plt.savefig(feature_plot_filename, dpi=300, bbox_inches='tight')
            plt.close()  # Close the figure
            
            if (i + 1) % 10 == 0:  # Print progress every 10 features
                print(f"Plotted {i + 1}/{n_features} features...")
        
        print(f"All {n_features} individual feature distribution plots saved to: {feature_plots_folder}")
    
    # Skip feature statistics plotting to save time
    print("Skipping feature statistics plotting to save time")
    # # Plot feature statistics summary
    # print("\n=== PLOTTING FEATURE STATISTICS SUMMARY ===")
    # 
    # # Calculate statistics for each feature
    # means = np.mean(all_data, axis=0)
    # stds = np.std(all_data, axis=0)
    # skewness = [stats.skew(all_data[:, i]) for i in range(all_data.shape[1])]
    # kurtosis = [stats.kurtosis(all_data[:, i]) for i in range(all_data.shape[1])]
    # 
    # # Create subplots
    # fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    # 
    # # Plot means
    # axes[0, 0].bar(range(len(means)), means, alpha=0.7, color='blue')
    # axes[0, 0].axhline(y=0, color='red', linestyle='--', linewidth=2, label='Target (0)')
    # axes[0, 0].set_title('Feature Means (should be ~0)')
    # axes[0, 0].set_xlabel('Feature Index')
    # axes[0, 0].set_ylabel('Mean')
    # axes[0, 0].legend()
    # axes[0, 0].grid(True, alpha=0.3)
    # 
    # # Plot standard deviations
    # axes[0, 1].bar(range(len(stds)), stds, alpha=0.7, color='green')
    # axes[0, 1].axhline(y=1, color='red', linestyle='--', linewidth=2, label='Target (1)')
    # axes[0, 1].set_title('Feature Standard Deviations (should be ~1)')
    # axes[0, 1].set_xlabel('Feature Index')
    # axes[0, 1].set_ylabel('Standard Deviation')
    # axes[0, 1].legend()
    # axes[0, 1].grid(True, alpha=0.3)
    # 
    # # Plot skewness
    # axes[1, 0].bar(range(len(skewness)), skewness, alpha=0.7, color='orange')
    # axes[1, 0].axhline(y=0, color='red', linestyle='--', linewidth=2, label='Normal (0)')
    # axes[1, 0].set_title('Feature Skewness (0 = symmetric)')
    # axes[1, 0].set_xlabel('Feature Index')
    # axes[1, 0].set_ylabel('Skewness')
    # axes[1, 0].legend()
    # axes[1, 0].grid(True, alpha=0.3)
    # 
    # # Plot kurtosis
    # axes[1, 1].bar(range(len(kurtosis)), kurtosis, alpha=0.7, color='purple')
    # axes[1, 1].axhline(y=0, color='red', linestyle='--', linewidth=2, label='Normal (0)')
    # axes[1, 1].set_title('Feature Kurtosis (0 = normal)')
    # axes[1, 1].set_xlabel('Feature Index')
    # axes[1, 1].set_ylabel('Kurtosis')
    # axes[1, 1].legend()
    # axes[1, 1].grid(True, alpha=0.3)
    # 
    # plt.tight_layout()
    # feature_stats_plot_filename = f'{folder}feature_statistics_summary.png'
    # plt.savefig(feature_stats_plot_filename, dpi=300, bbox_inches='tight')
    # plt.close()  # Close the figure instead of showing it
    # print(f"Feature statistics plot saved to: {feature_stats_plot_filename}")
    # 
    # # Print summary statistics
    # print(f"\n=== SUMMARY STATISTICS ===")
    # print(f"Mean of means: {np.mean(means):.6f} (should be ~0)")
    # print(f"Std of means: {np.std(means):.6f} (should be small)")
    # print(f"Mean of stds: {np.mean(stds):.6f} (should be ~1)")
    # print(f"Std of stds: {np.std(stds):.6f} (should be small)")
    # print(f"Mean skewness: {np.mean(skewness):.6f}")
    # print(f"Mean kurtosis: {np.mean(kurtosis):.6f}")
    
    # Print all input feature names
    print("\n=== INPUT FEATURE NAMES ===")
    for i, name in enumerate(final_column_names):
        print(f"{i:3d}: {name}")
    
    # Get additional feature indices (features to exclude from output)
    column_name_dict = {i: name for i, name in enumerate(final_column_names)}
    additional_feature_indices = get_additional_column_indices(column_name_dict)
    
    print(f"\nFeatures to exclude from output: {len(additional_feature_indices)}")
    print("Excluded features:")
    for idx in additional_feature_indices:
        print(f"  {idx}: {final_column_names[idx]}")
    
    # Calculate number of output features
    num_output_features = all_data.shape[1] - len(additional_feature_indices)
    print(f"\nNumber of output features: {num_output_features}")
    
    # Print output feature names
    output_feature_indices = [i for i in range(all_data.shape[1]) if i not in additional_feature_indices]
    print("\n=== OUTPUT FEATURE NAMES ===")
    for i, idx in enumerate(output_feature_indices):
        print(f"{i:3d}: {final_column_names[idx]}")
    
    # Generate sequences: sub-category data only for both training and validation (80/20 split)
    # Use the standardized data instead of all_data_combined
    train_data, val_data = generate_mixed_sequences(
        all_data_standardized, frame_indices_mapping, sequence_length, prediction_length, 
        original_sequence_length=original_sequence_length
    )
    
    return train_data, val_data, all_data.shape[1], num_output_features, additional_feature_indices, final_column_names, frame_indices_mapping, scaler

def generate_mixed_sequences(all_data_standardized, frame_indices_mapping, sequence_length, prediction_length, 
                           train_subsample_fraction=0.5, val_target_frames_only=True, original_sequence_length=480):
    """
    Generate sequences using only sub-category data for both training and validation:
    - Training: 80% of sub-category target frames
    - Validation: 20% of sub-category target frames
    - Adjusts starting positions based on sequence length difference
    """
    print(f"Generating sequences from sub-category data only:")
    print(f"  Training: 80% of target frames ({len(frame_indices_mapping)} starting points)")
    print(f"  Validation: 20% of target frames")
    print(f"Current sequence length: {sequence_length}")
    print(f"Original sequence length (used for category selection): {original_sequence_length}")
    
    # Calculate the offset needed to adjust starting positions
    sequence_offset = original_sequence_length - sequence_length
    print(f"Sequence offset adjustment: {sequence_offset} frames")
    
    train_sequences = []
    val_sequences = []
    
    # Set random seed for reproducible train/val split
    np.random.seed(42)
    
    # Generate sequences from target frames only, split 80/20
    print("Generating sequences from target frames with 80/20 train/val split...")
    for original_df_index, frame_index, local_df_index in frame_indices_mapping:
        # Get the standardized data for this dataframe
        standardized_data = all_data_standardized[local_df_index]
        
        # Adjust the starting position based on sequence length difference
        adjusted_start_index = frame_index + sequence_offset
        
        # Check if we have enough frames for the sequence
        if adjusted_start_index < 0 or adjusted_start_index + sequence_length + prediction_length > len(standardized_data):
            print(f"Warning: Not enough frames for sequence starting at adjusted index {adjusted_start_index} in dataframe {original_df_index}")
            continue
        
        # Extract the sequence from standardized data using adjusted starting position
        sequence = standardized_data[adjusted_start_index:adjusted_start_index + sequence_length + prediction_length]
        
        # Split into input and target
        input_sequence = sequence[:sequence_length]
        target_sequence = sequence[sequence_length:sequence_length + prediction_length]
        
        # Random train/val split (80% train, 20% val)
        if np.random.random() < 0.8:  # 80% for training
            train_sequences.append((input_sequence, target_sequence))
        else:  # 20% for validation
            val_sequences.append((input_sequence, target_sequence))
    
    print(f"Generated {len(train_sequences)} training sequences and {len(val_sequences)} validation sequences")
    print(f"Train/Val ratio: {len(train_sequences)/(len(train_sequences)+len(val_sequences)):.3f}")
    
    return train_sequences, val_sequences

def generate_hybrid_sequences(all_data_combined, frame_indices_mapping, sequence_length, prediction_length, 
                            subcategory_ratio=0.8, general_ratio=0.2):
    """
    Generate sequences using hybrid approach:
    - Training: 80% sub-category data + 20% general data
    - Validation: 80% sub-category data + 20% general data
    """
    print(f"Generating hybrid sequences:")
    print(f"  Sub-category data: {subcategory_ratio*100}%")
    print(f"  General data: {general_ratio*100}%")
    print(f"  Target frames: {len(frame_indices_mapping)} starting points")
    
    train_sequences = []
    val_sequences = []
    
    # Set random seed for reproducible results
    np.random.seed(42)
    
    # Calculate how many sequences we need from each source
    total_target_sequences = len(frame_indices_mapping)
    n_subcategory_train = int(total_target_sequences * subcategory_ratio * 0.8)  # 80% of subcategory for train
    n_subcategory_val = int(total_target_sequences * subcategory_ratio * 0.2)    # 20% of subcategory for val
    n_general_train = int(total_target_sequences * general_ratio * 0.8)          # 80% of general for train
    n_general_val = int(total_target_sequences * general_ratio * 0.2)            # 20% of general for val
    
    print(f"Target sequences: {n_subcategory_train} subcategory train, {n_subcategory_val} subcategory val")
    print(f"General sequences: {n_general_train} general train, {n_general_val} general val")
    
    # Generate sequences from sub-category target frames
    print("Generating sequences from sub-category target frames...")
    subcategory_sequences = []
    for original_df_index, frame_index, local_df_index in frame_indices_mapping:
        # Get the processed dataframe
        processed_data = all_data_combined[local_df_index]
        
        # Check if we have enough frames for the sequence
        if frame_index + sequence_length + prediction_length > len(processed_data):
            print(f"Warning: Not enough frames for sequence starting at frame {frame_index} in dataframe {original_df_index}")
            continue
        
        # Extract the sequence
        sequence = processed_data[frame_index:frame_index + sequence_length + prediction_length]
        
        # Split into input and target
        input_sequence = sequence[:sequence_length]
        target_sequence = sequence[sequence_length:sequence_length + prediction_length]
        
        subcategory_sequences.append((input_sequence, target_sequence))
    
    # Randomly split sub-category sequences into train/val
    np.random.shuffle(subcategory_sequences)
    train_sequences.extend(subcategory_sequences[:n_subcategory_train])
    val_sequences.extend(subcategory_sequences[n_subcategory_train:n_subcategory_train + n_subcategory_val])
    
    # Generate sequences from general data (all dataframes, random sampling)
    print("Generating sequences from general data...")
    general_sequences = []
    for local_df_index, processed_data in enumerate(all_data_combined):
        # Calculate how many sequences we can extract from this dataframe
        max_start_idx = len(processed_data) - sequence_length - prediction_length
        if max_start_idx <= 0:
            continue
            
        # Sample some sequences from this dataframe
        n_sequences_from_df = min(10, max_start_idx)  # Limit per dataframe to avoid imbalance
        start_indices = np.random.choice(max_start_idx, n_sequences_from_df, replace=False)
        
        for start_idx in start_indices:
            # Extract the sequence
            sequence = processed_data[start_idx:start_idx + sequence_length + prediction_length]
            
            # Split into input and target
            input_sequence = sequence[:sequence_length]
            target_sequence = sequence[sequence_length:sequence_length + prediction_length]
            
            general_sequences.append((input_sequence, target_sequence))
    
    # Randomly split general sequences into train/val
    np.random.shuffle(general_sequences)
    train_sequences.extend(general_sequences[:n_general_train])
    val_sequences.extend(general_sequences[n_general_train:n_general_train + n_general_val])
    
    # Shuffle the final sequences
    np.random.shuffle(train_sequences)
    np.random.shuffle(val_sequences)
    
    print(f"Generated {len(train_sequences)} training sequences and {len(val_sequences)} validation sequences")
    print(f"Train/Val ratio: {len(train_sequences)/(len(train_sequences)+len(val_sequences)):.3f}")
    print(f"Sub-category in train: {n_subcategory_train}/{len(train_sequences)} ({n_subcategory_train/len(train_sequences)*100:.1f}%)")
    print(f"Sub-category in val: {n_subcategory_val}/{len(val_sequences)} ({n_subcategory_val/len(val_sequences)*100:.1f}%)")
    
    return train_sequences, val_sequences

def resume_training_from_checkpoint(checkpoint_path, train_data, val_data, sequence_length, prediction_length, 
                                  model_params, additional_feature_indices, desired_categories, folder,
                                  initial_epoch=0, epochs=300):
    """
    Resume training from a previous checkpoint
    """
    print(f"Resuming training from checkpoint: {checkpoint_path}")
    
    # Load the model from checkpoint
    model = load_model(checkpoint_path)
    
    
    # Recompile with current learning rate schedule and gradient clipping
    model.compile(
        optimizer=Adam(learning_rate=1e-4, clipnorm=1.0), 
        loss='mean_squared_error'
    )
    
    print(f"Loaded model with {model.count_params()} parameters")
    print(f"Resuming from epoch {initial_epoch}")
    
    # Create data generators
    train_gen = data_generator_from_sequences(
        train_data, sequence_length, prediction_length, batch_size, random_seed, additional_feature_indices
    )
    val_gen = data_generator_from_sequences(
        val_data, sequence_length, prediction_length, batch_size, random_seed, additional_feature_indices
    )
    
    # Calculate steps per epoch
    train_steps = len(train_data) // batch_size
    val_steps = len(val_data) // batch_size
    
    # Callbacks (same as original training)
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1),
        ModelCheckpoint(
                filepath=f'{folder}best_model_relative_features_with_depth_{"Bottom_Playing_and_Alone"}_decoder_fixed_v4_enhanced_standardized_no_velocity_test_depth_output_seq{sequence_length}_latent{latent_dims[0]}_bottleneck{bottleneck_dims[0]}.h5',
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False
        ),
        CSVLogger(
            filename=f'{folder}training_log_resumed_{"Bottom_Playing_and_Alone"}_relative_with_depth_v4_enhanced_standardized_no_velocity_test_depth_output_seq{sequence_length}_latent{latent_dims[0]}_bottleneck{bottleneck_dims[0]}.csv',
            separator=',',
            append=False
        )
    ]
    
    # Resume training
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        steps_per_epoch=train_steps,
        validation_steps=val_steps,
        epochs=epochs,
        initial_epoch=initial_epoch,
        callbacks=callbacks,
        verbose=1
    )
    
    return model, history

def generate_sequences_from_target_frames(all_data_combined, frame_indices_mapping, sequence_length, prediction_length, test_size=0.3):
    """
    Generate sequences using target frames as starting points and split train/val randomly
    """
    print(f"Generating sequences from {len(frame_indices_mapping)} target frame starting points")
    
    train_sequences = []
    val_sequences = []
    
    # Set random seed for reproducible train/val split
    np.random.seed(42)
    
    for original_df_index, frame_index, local_df_index in frame_indices_mapping:
        # Get the processed dataframe
        processed_data = all_data_combined[local_df_index]
        
        # Check if we have enough frames for the sequence
        if frame_index + sequence_length + prediction_length > len(processed_data):
            print(f"Warning: Not enough frames for sequence starting at frame {frame_index} in dataframe {original_df_index}")
            continue
        
        # Extract the sequence
        sequence = processed_data[frame_index:frame_index + sequence_length + prediction_length]
        
        # Split into input and target
        input_sequence = sequence[:sequence_length]
        target_sequence = sequence[sequence_length:sequence_length + prediction_length]
        
        # Random train/val split
        if np.random.random() < (1 - test_size):  # 70% train, 30% val
            train_sequences.append((input_sequence, target_sequence))
        else:
            val_sequences.append((input_sequence, target_sequence))
    
    print(f"Generated {len(train_sequences)} training sequences and {len(val_sequences)} validation sequences")
    
    return train_sequences, val_sequences

def create_attention_model(sequence_length, prediction_length, num_input_features, num_output_features, 
                         latent_dim, dropout_rate, regularization_lambda, bottleneck_dim):
    """
    Create enhanced attention-based model with increased capacity for sub-category discovery
    """
    print("Creating enhanced attention model with increased capacity...")
    
    # Encoder - Enhanced with more layers
    encoder_inputs = Input(shape=(sequence_length, num_input_features))
    
    # First LSTM layer
    encoder_lstm1 = LSTM(latent_dim, return_sequences=True,
                        return_state=True, kernel_regularizer=l2(regularization_lambda))
    encoder_outputs1, state_h1, state_c1 = encoder_lstm1(encoder_inputs)
    encoder_outputs1 = Dropout(dropout_rate)(encoder_outputs1)
    
    # Second LSTM layer for more capacity
    encoder_lstm2 = LSTM(latent_dim, return_sequences=True,
                        return_state=True, kernel_regularizer=l2(regularization_lambda))
    encoder_outputs, state_h, state_c = encoder_lstm2(encoder_outputs1)
    encoder_outputs = Dropout(dropout_rate)(encoder_outputs)

    # Enhanced Bottleneck Layer with more capacity
    bottleneck1 = Dense(bottleneck_dim, activation='relu')(encoder_outputs)
    bottleneck1 = Dropout(dropout_rate)(bottleneck1)
    
    bottleneck2 = Dense(bottleneck_dim, activation='relu')(bottleneck1)
    bottleneck = Dropout(dropout_rate)(bottleneck2)

    # Multi-Head Attention with more heads for better sub-category discrimination
    attention_output = MultiHeadAttention(
        num_heads=8, key_dim=bottleneck_dim//2)(bottleneck, bottleneck)  # More heads, smaller key_dim
    attention_output = LayerNormalization(epsilon=1e-6)(attention_output)
    attention_output = Dropout(dropout_rate)(attention_output)

    # Decoder - Enhanced with more layers
    decoder_inputs = Input(shape=(prediction_length, num_input_features))
    
    # First decoder LSTM layer
    decoder_lstm1 = LSTM(latent_dim, return_sequences=True,
                        return_state=True, kernel_regularizer=l2(regularization_lambda))
    decoder_outputs1, _, _ = decoder_lstm1(decoder_inputs, initial_state=[state_h, state_c])
    decoder_outputs1 = Dropout(dropout_rate)(decoder_outputs1)
    
    # Second decoder LSTM layer
    decoder_lstm2 = LSTM(latent_dim, return_sequences=True,
                        return_state=True, kernel_regularizer=l2(regularization_lambda))
    decoder_outputs, _, _ = decoder_lstm2(decoder_outputs1)
    decoder_outputs = Dropout(dropout_rate)(decoder_outputs)

    # Enhanced Multi-Head Attention with more heads
    decoder_attention_output = MultiHeadAttention(
        num_heads=8, key_dim=bottleneck_dim//2)(decoder_outputs, attention_output)
    decoder_attention_output = LayerNormalization(epsilon=1e-6)(decoder_attention_output)
    decoder_attention_output = Dropout(dropout_rate)(decoder_attention_output)

    # Concatenate attention output and decoder LSTM output
    concat_layer = Concatenate(axis=-1)([decoder_outputs, decoder_attention_output])
    concat_layer = Dropout(dropout_rate)(concat_layer)

    # Enhanced output layer with more capacity
    hidden_layer = TimeDistributed(Dense(bottleneck_dim, activation='relu'))(concat_layer)
    hidden_layer = Dropout(dropout_rate)(hidden_layer)
    
    output_layer = TimeDistributed(
        Dense(num_output_features, activation='linear'))(hidden_layer)

    # Define the model
    model = Model([encoder_inputs, decoder_inputs], output_layer)
    
    print(f"Enhanced model created with {model.count_params()} parameters")
    return model

def train_model(train_data, val_data, sequence_length, prediction_length, model_params, additional_feature_indices, desired_categories):
    """
    Train the model using data generators
    """
    print("Starting model training...")
    
    # Create model
    model = create_attention_model(**model_params)
    
    # Compile model with gradient clipping and better regularization
    model.compile(
        optimizer=Adam(learning_rate=1e-4, clipnorm=1.0), 
        loss='mean_squared_error'
    )
    
    # Print model summary
    print("\nModel Summary:")
    model.summary()
    print(f"\nEncoder input shape: (batch_size, {model_params['sequence_length']}, {model_params['num_input_features']})")
    print(f"Decoder input shape: (batch_size, {model_params['prediction_length']}, {model_params['num_input_features']})")
    print(f"Output shape: (batch_size, {model_params['prediction_length']}, {model_params['num_output_features']})")
    print(f"Additional features excluded from output: {len(additional_feature_indices)}")
    
    # Create data generators
    train_gen = data_generator_from_sequences(
        train_data, sequence_length, prediction_length, batch_size, random_seed, additional_feature_indices
    )
    val_gen = data_generator_from_sequences(
        val_data, sequence_length, prediction_length, batch_size, random_seed, additional_feature_indices
    )
    
    # Calculate steps per epoch
    train_steps = len(train_data) // batch_size
    val_steps = len(val_data) // batch_size
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1),
        ModelCheckpoint(
                filepath=f'{folder}best_model_relative_features_with_depth_{"Bottom_Playing_and_Alone"}_decoder_fixed_v4_enhanced_standardized_no_velocity_test_depth_output_seq{sequence_length}_latent{latent_dims[0]}_bottleneck{bottleneck_dims[0]}.h5',
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False
        ),
        CSVLogger(
            filename=f'{folder}training_log_{"Bottom_Playing_and_Alone"}_relative_with_depth_v4_enhanced_standardized_no_velocity_test_depth_output_seq{sequence_length}_latent{latent_dims[0]}_bottleneck{bottleneck_dims[0]}.csv',
            separator=',',
            append=False
        )
    ]
    
    # Train model
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        steps_per_epoch=train_steps,
        validation_steps=val_steps,
        epochs=300,
        callbacks=callbacks,
        verbose=1
    )
    
    return model, history

def main():
    """
    Main training function
    """
    print("Loading data...")
    all_dfs = pd.read_pickle("/media/fenglab/Seagate Hub/Shank3_macaque/second cohort/social/limb_model/all_dfs_with_limb_features.pkl")
    
    # Load the behavioral category data
    category_file = '/media/fenglab/New Volume/River_data_backup/2nd_cohort_social/3d/model/data_for_analysis_200_1_5_64_clusters.csv'
    print(f"Loading behavioral category data from: {category_file}")
    category_df = pd.read_csv(category_file)
    print(f"Loaded category data with shape: {category_df.shape}")
    
    # Define categories to train (both categories together)
    desired_categories = ['Bottom Playing', 'Bottom Alone']
    
    print(f"\n{'='*60}")
    print(f"TRAINING MODEL FOR CATEGORIES: {desired_categories}")
    print(f"{'='*60}")
    
    # Train model for both categories
    train_model_for_categories(all_dfs, category_df, desired_categories)

def train_model_for_categories(all_dfs, category_df, desired_categories):
    """
    Train model for specific categories
    """
    # Filter for desired categories
    filtered_df = category_df[category_df['Category'].isin(desired_categories)]
    print(f"Filtered to {len(filtered_df)} frames with categories: {desired_categories}")
    
    # We don't need a dictionary - we can directly use DataFrame Index and Frame Index from the category file
    print(f"Will process {len(filtered_df)} frames with categories: {desired_categories}")
    
    # Data subsampling parameters (same as in latent extraction script)
    use_data_subset = False
    data_subset_fraction = 0.3  # Use 30% of dataframes (same as training)
    
    if use_data_subset:
        # Generate subset (same as training with same random seed)
        print(f"Using data subset: {data_subset_fraction*100}% of dataframes")
        n_subset = int(len(all_dfs) * data_subset_fraction)
        subset_indices = np.random.choice(len(all_dfs), n_subset, replace=False)
        all_dfs = [all_dfs[i] for i in subset_indices]
        # Keep track of original dataframe indices
        original_df_indices = subset_indices
        print(f"Selected {len(all_dfs)} dataframes out of original {len(all_dfs)/data_subset_fraction:.0f}")
        print(f"Selected dataframe indices: {sorted(subset_indices)}")
        print(f"Range of selected indices: {min(subset_indices)} to {max(subset_indices)}")
        
        # Save the subset indices for reproducibility
        # subset_indices_save_path = '/media/fenglab/Seagate Hub/Shank3_macaque/second cohort/social/limb_model/relative_features/dataframe_subset_indices.npy'
        # os.makedirs(os.path.dirname(subset_indices_save_path), exist_ok=True)
        #np.save(subset_indices_save_path, subset_indices)
        #print(f"Saved dataframe subset indices to: {subset_indices_save_path}")
    else:
        # Use all dataframes
        original_df_indices = np.arange(len(all_dfs))
    
    # Model parameters
    sequence_length = sequence_lengths[0]
    prediction_length = prediction_lengths[0]
    latent_dim = latent_dims[0]
    dropout_rate = dropout_rates[0]
    regularization_lambda = regularization_lambdas[0]
    bottleneck_dim = bottleneck_dims[0]
    
    # Preprocess data using subset function
    train_data, val_data, num_input_features, num_output_features, additional_feature_indices, feature_names, frame_indices_mapping, scaler = preprocess_data_with_relative_features_subset(
        all_dfs, sequence_length, prediction_length, filtered_df, original_df_indices
    )
    
    # Update model parameters
    model_params = {
        'sequence_length': sequence_length,
        'prediction_length': prediction_length,
        'num_input_features': num_input_features,
        'num_output_features': num_output_features,
        'latent_dim': latent_dim,
        'dropout_rate': dropout_rate,
        'regularization_lambda': regularization_lambda,
        'bottleneck_dim': bottleneck_dim
    }
    
    # Print model summary before training
    print("\n=== MODEL SUMMARY ===")
    model, history = train_model(train_data, val_data, sequence_length, prediction_length, model_params, additional_feature_indices, desired_categories)
    
    # Save final model with category information in filename
    model_filename = f'{folder}final_model_relative_features_with_depth_{"Bottom_Playing_and_Alone"}_decoder_fixed_v4_enhanced_standardized_no_velocity_test_depth_output_seq48.h5'
    model.save(model_filename)
    print(f"Model saved to {model_filename}")
    
    # Save feature information
    feature_info = {
        'input_features': num_input_features,
        'output_features': num_output_features,
        'feature_names': feature_names,
        'additional_feature_indices': additional_feature_indices,
        'desired_categories': desired_categories,
        'frame_indices_mapping': frame_indices_mapping,
        'scaler': scaler  # Save the fitted scaler for later use
    }
    feature_info_filename = f'{folder}feature_info_relative_features_with_depth_{"Bottom_Playing_and_Alone"}_v4_enhanced_standardized_no_velocity_test_depth_output_seq48.pkl'
    with open(feature_info_filename, 'wb') as f:
        pickle.dump(feature_info, f)
    print(f"Feature info saved to {feature_info_filename}")
    
    # Save frame indices mapping for reference
    mapping_filename = f'{folder}frame_indices_mapping_{"Bottom_Playing_and_Alone"}_relative_with_depth_v4_enhanced_standardized_no_velocity_test_depth_output_seq48.pkl'
    with open(mapping_filename, 'wb') as f:
        pickle.dump(frame_indices_mapping, f)
    print(f"Frame indices mapping saved to {mapping_filename}")
    
    # Skip plotting training history to save time
    print("Skipping training history plotting to save time")
    # plt.figure(figsize=(12, 4))
    # 
    # plt.subplot(1, 2, 1)
    # plt.plot(history.history['loss'], label='Training Loss')
    # plt.plot(history.history['val_loss'], label='Validation Loss')
    # plt.title('Model Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.legend()
    # 
    # plt.subplot(1, 2, 2)
    # plt.plot(history.history['mae'], label='Training MAE')
    # plt.plot(history.history['val_mae'], label='Validation MAE')
    # plt.title('Model MAE')
    # plt.xlabel('Epoch')
    # plt.ylabel('MAE')
    # plt.legend()
    # 
    # plt.tight_layout()
    # plot_filename = f'{folder}training_history_absolute_features_{"Bottom_Playing_and_Alone"}_v4_enhanced_standardized_no_probe_output_seq48.png'
    # plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    # plt.show()
    
    print("Training completed!")

def resume_training_example():
    """
    Example of how to resume training from a checkpoint
    """
    # Use the same folder as defined in the script
    folder = '/home/fenglab/Documents/dlc/attention_model_scripts/model_2_sub_train_with_sub/'
    desired_categories = ['Bottom Playing', 'Bottom Alone']
    
    # Path to your saved model checkpoint (matches the actual checkpoint path from training)
    sequence_length = sequence_lengths[0]
    latent_dim = latent_dims[0]
    bottleneck_dim = bottleneck_dims[0]
    checkpoint_path = f'{folder}best_model_relative_features_with_depth_{"Bottom_Playing_and_Alone"}_decoder_fixed_v4_enhanced_standardized_no_velocity_test_depth_output_seq{sequence_length}_latent{latent_dim}_bottleneck{bottleneck_dim}.h5'
    
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at: {checkpoint_path}")
        print("Please run the main training first to create a checkpoint.")
        return
    
    print(f"Found checkpoint at: {checkpoint_path}")
    print("To determine the correct initial_epoch, check the training logs or CSV files.")
    print("Look for files like 'training_history_*.csv' to see how many epochs were completed.")
    
    # Load the same data as in main()
    print("Loading data for resume training...")
    all_dfs = pd.read_pickle("/media/fenglab/Seagate Hub/Shank3_macaque/second cohort/social/limb_model/all_dfs_with_limb_features.pkl")
    
    # Load the behavioral category data
    category_file = '/media/fenglab/New Volume/River_data_backup/2nd_cohort_social/3d/model/data_for_analysis_200_1_5_64_clusters.csv'
    category_df = pd.read_csv(category_file)
    filtered_df = category_df[category_df['Category'].isin(desired_categories)]
    
    # Use all dataframes for training (no subset)
    original_df_indices = np.arange(len(all_dfs))
    
    # Model parameters (use global parameters to ensure consistency)
    sequence_length = sequence_lengths[0]
    prediction_length = prediction_lengths[0]
    latent_dim = latent_dims[0]
    dropout_rate = dropout_rates[0]
    regularization_lambda = regularization_lambdas[0]
    bottleneck_dim = bottleneck_dims[0]
    
    # Preprocess data
    train_data, val_data, num_input_features, num_output_features, additional_feature_indices, feature_names, frame_indices_mapping, scaler = preprocess_data_with_relative_features_subset(
        all_dfs, sequence_length, prediction_length, filtered_df, original_df_indices
    )
    
    # Model parameters
    model_params = {
        'sequence_length': sequence_length,
        'prediction_length': prediction_length,
        'num_input_features': num_input_features,
        'num_output_features': num_output_features,
        'latent_dim': latent_dim,
        'dropout_rate': dropout_rate,
        'regularization_lambda': regularization_lambda,
        'bottleneck_dim': bottleneck_dim
    }
    
    # Resume training
    # Note: If you want to resume from a specific epoch, change initial_epoch to that number
    # For example, if previous training stopped at epoch 150, set initial_epoch=150
    model, history = resume_training_from_checkpoint(
        checkpoint_path, train_data, val_data, sequence_length, prediction_length,
        model_params, additional_feature_indices, desired_categories, folder,
        initial_epoch=0,  # Set to the epoch where you want to resume (0 if starting from beginning)
        epochs=600  # Total number of epochs to train for
    )
    
    print("Resume training completed!")

if __name__ == "__main__":
    # Uncomment the line below to resume training instead of starting fresh
    # resume_training_example()
    main() 