#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline script that reads configuration from config.yaml
Runs the complete attention model workflow:
1. Model training (model_training.py)
2. Latent extraction (latent_extraction.py)
3. PCA analysis (pca.py)
4. t-SNE analysis (tsne.py)
"""

import sys
import os
import importlib.util
import yaml

# ============================================================================
# LOAD CONFIGURATION
# ============================================================================

def load_config(config_path='config.yaml'):
    """Load configuration from YAML file"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(script_dir, config_path)
    
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def import_module_from_file(module_name, file_path):
    """Import a Python module from a file path"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

def update_module_params(module, params):
    """Update module-level variables with new parameter values"""
    for key, value in params.items():
        if hasattr(module, key):
            setattr(module, key, value)
            print(f"  Updated {key} = {value}")
        else:
            print(f"  Warning: {key} not found in module, skipping")

def get_category_string(categories):
    """Get category string for file naming"""
    return "Bottom_Playing_and_Alone"  # Fixed naming convention

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Run the complete pipeline"""
    # Load configuration
    print("Loading configuration from config.yaml...")
    config = load_config()
    
    # Extract configuration sections
    model_cfg = config['model']
    training_cfg = config['training']
    categories_cfg = config['categories']
    paths_cfg = config['paths']
    pca_cfg = config['pca']
    tsne_cfg = config['tsne']
    pipeline_cfg = config['pipeline']
    
    print("="*80)
    print("ATTENTION MODEL PIPELINE (Config-based)")
    print("="*80)
    print("\nConfiguration:")
    print(f"  Sequence Length: {model_cfg['sequence_length']}")
    print(f"  Prediction Length: {model_cfg['prediction_length']}")
    print(f"  Latent Dimension: {model_cfg['latent_dim']}")
    print(f"  Bottleneck Dimension: {model_cfg['bottleneck_dim']}")
    print(f"  Categories: {categories_cfg['desired_categories']}")
    print("="*80)
    
    # Get the directory of this script
    pipeline_dir = os.path.dirname(os.path.abspath(__file__))
    
    # ========================================================================
    # STEP 1: Model Training
    # ========================================================================
    if pipeline_cfg['run_training']:
        print("\n" + "="*80)
        print("STEP 1: MODEL TRAINING")
        print("="*80)
        
        training_script = os.path.join(pipeline_dir, 'model_training.py')
        print(f"Importing: {training_script}")
        
        # Import the module
        model_training = import_module_from_file('model_training', training_script)
        
        # Update parameters
        print("Updating parameters...")
        update_module_params(model_training, {
            'sequence_lengths': [model_cfg['sequence_length']],
            'prediction_lengths': [model_cfg['prediction_length']],
            'latent_dims': [model_cfg['latent_dim']],
            'bottleneck_dims': [model_cfg['bottleneck_dim']],
            'original_sequence_length': model_cfg['original_sequence_length'],
            'include_depth_features': model_cfg['include_depth_features'],
            'dropout_rates': [training_cfg['dropout_rate']],
            'regularization_lambdas': [training_cfg['regularization_lambda']],
            'batch_size': training_cfg['batch_size'],
            'patience': training_cfg['patience'],
            'test_size': training_cfg['test_size'],
            'random_seed': training_cfg['random_seed'],
            'folder': paths_cfg['model_output']
        })
        
        # Run training
        print("\nStarting model training...")
        model_training.main()
        
        print("\n✅ Model training completed!")
    else:
        print("\n⏭️  Skipping model training")
    
    # ========================================================================
    # STEP 2: Latent Space Extraction
    # ========================================================================
    if pipeline_cfg['run_latent_extraction']:
        print("\n" + "="*80)
        print("STEP 2: LATENT SPACE EXTRACTION")
        print("="*80)
        
        latent_script = os.path.join(pipeline_dir, 'latent_extraction.py')
        print(f"Importing: {latent_script}")
        
        # Import the module
        latent_extraction = import_module_from_file('latent_extraction', latent_script)
        
        # Construct paths for latent extraction
        model_folder = paths_cfg['model_output']
        latent_output_folder = paths_cfg['latent_output']
        
        # Update parameters
        print("Updating parameters...")
        update_module_params(latent_extraction, {
            'sequence_length': model_cfg['sequence_length'],
            'prediction_length': model_cfg['prediction_length'],
            'latent_dim': model_cfg['latent_dim'],
            'bottleneck_dim': model_cfg['bottleneck_dim'],
            'original_sequence_length': model_cfg['original_sequence_length'],
            'include_depth_features': model_cfg['include_depth_features'],
            'batch_size': training_cfg['batch_size'],
            'random_seed': training_cfg['random_seed'],
            'desired_categories': categories_cfg['desired_categories'],
            'model_folder': model_folder,
            'latent_output_folder': latent_output_folder
        })
        
        # Run latent extraction
        print("\nStarting latent extraction...")
        latent_extraction.main()
        
        print("\n✅ Latent extraction completed!")
    else:
        print("\n⏭️  Skipping latent extraction")
    
    # ========================================================================
    # STEP 3: PCA Analysis
    # ========================================================================
    if pipeline_cfg['run_pca']:
        print("\n" + "="*80)
        print("STEP 3: PCA ANALYSIS")
        print("="*80)
        
        pca_script = os.path.join(pipeline_dir, 'pca.py')
        print(f"Importing: {pca_script}")
        
        # Import the module
        pca_module = import_module_from_file('pca_module', pca_script)
        
        # Construct paths for PCA
        latent_input_folder = paths_cfg['latent_output']
        pca_output_folder = paths_cfg['pca_output']
        
        # Update parameters
        print("Updating parameters...")
        update_module_params(pca_module, {
            'sequence_length': model_cfg['sequence_length'],
            'latent_dim': model_cfg['latent_dim'],
            'bottleneck_dim': model_cfg['bottleneck_dim'],
            'desired_categories': categories_cfg['desired_categories'],
            'latent_input_folder': latent_input_folder,
            'pca_output_folder': pca_output_folder
        })
        
        # Run PCA
        print("\nStarting PCA analysis...")
        pca_module.main()
        
        print("\n✅ PCA analysis completed!")
    else:
        print("\n⏭️  Skipping PCA analysis")
    
    # ========================================================================
    # STEP 4: t-SNE Analysis
    # ========================================================================
    if pipeline_cfg['run_tsne']:
        print("\n" + "="*80)
        print("STEP 4: t-SNE ANALYSIS")
        print("="*80)
        
        tsne_script = os.path.join(pipeline_dir, 'tsne.py')
        print(f"Importing: {tsne_script}")
        
        # Import the module
        tsne_module = import_module_from_file('tsne_module', tsne_script)
        
        # Construct paths for t-SNE
        pca_input_folder = paths_cfg['pca_output']
        latent_input_folder = paths_cfg['latent_output']
        tsne_output_folder = paths_cfg['tsne_output']
        
        # Update parameters
        print("Updating parameters...")
        update_module_params(tsne_module, {
            'sequence_length': model_cfg['sequence_length'],
            'latent_dim': model_cfg['latent_dim'],
            'bottleneck_dim': model_cfg['bottleneck_dim'],
            'desired_categories': categories_cfg['desired_categories'],
            'n_batches': tsne_cfg['n_batches'],
            'n_clusters': tsne_cfg['n_clusters'],
            'perplexity': tsne_cfg['perplexity'],
            'pca_input_folder': pca_input_folder,
            'latent_input_folder': latent_input_folder,
            'tsne_output_folder': tsne_output_folder
        })
        
        # Run t-SNE
        print("\nStarting t-SNE analysis...")
        tsne_module.main()
        
        print("\n✅ t-SNE analysis completed!")
    else:
        print("\n⏭️  Skipping t-SNE analysis")
    
    # ========================================================================
    # Pipeline Complete
    # ========================================================================
    print("\n" + "="*80)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("\nResults saved to:")
    if pipeline_cfg['run_training']:
        print(f"  Models: {paths_cfg['model_output']}")
    if pipeline_cfg['run_latent_extraction']:
        latent_dir = f"{paths_cfg['latent_output']}bottom_playing_and_alone_latent_seq_{model_cfg['sequence_length']}_latent_{model_cfg['latent_dim']}_bottleneck_{model_cfg['bottleneck_dim']}_no_velocity_test_depth/"
        print(f"  Latent: {latent_dir}")
    if pipeline_cfg['run_pca']:
        pca_dir = f"{paths_cfg['pca_output']}bottom_playing_and_alone_pca_seq_{model_cfg['sequence_length']}_latent_{model_cfg['latent_dim']}_bottleneck_{model_cfg['bottleneck_dim']}_no_velocity_test_depth/"
        print(f"  PCA: {pca_dir}")
    if pipeline_cfg['run_tsne']:
        tsne_dir = f"{paths_cfg['tsne_output']}bottom_playing_and_alone_pca_seq_{model_cfg['sequence_length']}_latent_{model_cfg['latent_dim']}_bottleneck_{model_cfg['bottleneck_dim']}_no_velocity_test_depth/"
        print(f"  t-SNE: {tsne_dir}")
    print("="*80)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

