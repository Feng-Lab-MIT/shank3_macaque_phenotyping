# Pipeline Architecture - How It Works

## Overview

The pipeline now properly reads ALL settings from `config.yaml` and passes them to individual scripts. No more hardcoded paths or parameters in the scripts!

## Data Flow

```
config.yaml
    ↓
run_pipeline_from_config.py (reads config, passes to scripts)
    ↓
Individual scripts (receive parameters, don't construct paths)
```

## How Configuration Works

### 1. config.yaml - Single Source of Truth

All settings are defined here:

```yaml
model:
  sequence_length: 240
  latent_dim: 64
  bottleneck_dim: 32

paths:
  model_output: "/media/fenglab/newssd/social/model_output/"
  latent_output: "/media/fenglab/newssd/social/latent_output/"
  pca_output: "/media/fenglab/newssd/social/pca_output/"
  tsne_output: "/media/fenglab/newssd/social/tsne_output/"
```

### 2. Pipeline Script - Reads and Distributes

`run_pipeline_from_config.py` does:

1. **Loads config.yaml**
   ```python
   config = load_config('config.yaml')
   model_cfg = config['model']
   paths_cfg = config['paths']
   ```

2. **Imports each script**
   ```python
   latent_extraction = import_module_from_file('latent_extraction', 'latent_extraction.py')
   ```

3. **Updates script parameters**
   ```python
   update_module_params(latent_extraction, {
       'sequence_length': model_cfg['sequence_length'],
       'model_folder': paths_cfg['model_output'],
       'latent_output_folder': paths_cfg['latent_output']
   })
   ```

4. **Runs the script**
   ```python
   latent_extraction.main()
   ```

### 3. Individual Scripts - Use What They're Given

Each script now:

1. **Declares default folder variables** (used when run standalone)
   ```python
   # In latent_extraction.py
   model_folder = '/home/fenglab/Documents/dlc/...'  # Default
   latent_output_folder = '/media/fenglab/newssd/social/'  # Default
   ```

2. **Uses these variables in main()**
   ```python
   def main():
       # Construct specific paths using provided folders
       model_path = f'{model_folder}best_model_...seq{sequence_length}...'
       output_dir = f'{latent_output_folder}bottom_playing_...'
   ```

3. **Pipeline can override defaults**
   ```python
   # When run through pipeline:
   latent_extraction.model_folder = '/custom/path/'  # Override!
   latent_extraction.main()  # Uses /custom/path/
   ```

## Path Construction Pattern

Each script receives **base folders** from config and constructs **specific subdirectories**:

### Example: Latent Extraction

**Config provides:**
```yaml
latent_output: "/media/fenglab/newssd/social/latent_output/"
```

**Script constructs:**
```python
output_dir = f'{latent_output_folder}bottom_playing_and_alone_latent_seq_{sequence_length}_latent_{latent_dim}_bottleneck_{bottleneck_dim}_no_velocity_test_depth/'
# Result: /media/fenglab/newssd/social/latent_output/bottom_playing_and_alone_latent_seq_240_latent_64_bottleneck_32_no_velocity_test_depth/
```

This pattern allows:
- ✅ Clean base paths in config
- ✅ Automatic subdirectory naming based on parameters
- ✅ Easy path changes (just edit config)

## Parameter Flow Diagram

```
config.yaml
    model:
        sequence_length: 240
        latent_dim: 64
    paths:
        latent_output: "/media/.../latent_output/"
            ↓
run_pipeline_from_config.py
    Read config
    Extract: sequence_length=240, latent_dim=64
    Extract: latent_output_folder="/media/.../latent_output/"
            ↓
    Import latent_extraction.py
    Update: latent_extraction.sequence_length = 240
    Update: latent_extraction.latent_dim = 64
    Update: latent_extraction.latent_output_folder = "/media/.../latent_output/"
            ↓
    Call: latent_extraction.main()
            ↓
latent_extraction.py main()
    Construct: output_dir = f'{latent_output_folder}...seq_{sequence_length}_latent_{latent_dim}...'
    Result: "/media/.../latent_output/...seq_240_latent_64..."
```

## What Each Script Receives

### Model Training
```python
# From config
sequence_lengths = [config['model']['sequence_length']]
latent_dims = [config['model']['latent_dim']]
folder = config['paths']['model_output']
```

### Latent Extraction
```python
# From config
sequence_length = config['model']['sequence_length']
latent_dim = config['model']['latent_dim']
model_folder = config['paths']['model_output']
latent_output_folder = config['paths']['latent_output']
```

### PCA
```python
# From config
sequence_length = config['model']['sequence_length']
latent_dim = config['model']['latent_dim']
latent_input_folder = config['paths']['latent_output']
pca_output_folder = config['paths']['pca_output']
```

### t-SNE
```python
# From config
sequence_length = config['model']['sequence_length']
latent_dim = config['model']['latent_dim']
n_batches = config['tsne']['n_batches']
n_clusters = config['tsne']['n_clusters']
pca_input_folder = config['paths']['pca_output']
latent_input_folder = config['paths']['latent_output']
tsne_output_folder = config['paths']['tsne_output']
```

## Benefits of This Architecture

### ✅ Single Source of Truth
- All settings in one place (`config.yaml`)
- No hunting through scripts to find hardcoded values

### ✅ Easy to Change
Want different output folders? Just edit config:
```yaml
paths:
  latent_output: "/new/path/here/"
```

### ✅ Consistent Parameters
Pipeline ensures all scripts use the same values:
- No more `latent_dim=64` in one script and `latent_dim=512` in another

### ✅ Backwards Compatible
Scripts can still run standalone:
```bash
# Standalone - uses defaults in script
python latent_extraction.py

# Through pipeline - uses config
python run_pipeline_from_config.py
```

### ✅ Flexible
Can skip steps easily:
```yaml
pipeline:
  run_training: false      # Skip
  run_latent_extraction: true
  run_pca: true
  run_tsne: true
```

## Example Usage

### Quick Run with Defaults
```bash
python run_pipeline_from_config.py
```

### Custom Configuration
```yaml
# config.yaml
model:
  sequence_length: 480  # Longer sequences
  latent_dim: 128      # Bigger model

paths:
  model_output: "/custom/models/"
  latent_output: "/custom/latent/"
  pca_output: "/custom/pca/"
  tsne_output: "/custom/tsne/"
```

```bash
python run_pipeline_from_config.py
```

All scripts will automatically use:
- `sequence_length=480`
- `latent_dim=128`
- Custom output paths

## File Organization

With the config paths, your files will be organized like:

```
/media/fenglab/newssd/social/
├── model_output/
│   └── best_model_...seq240_latent64_bottleneck32.h5
├── latent_output/
│   └── bottom_playing_and_alone_latent_seq_240_latent_64_bottleneck_32_no_velocity_test_depth/
│       ├── latent_representations_0.npy
│       └── indices_0.npy
├── pca_output/
│   └── bottom_playing_and_alone_pca_seq_240_latent_64_bottleneck_32_no_velocity_test_depth/
│       └── pca_results/
│           ├── pca_vectors_...npy
│           └── pca_model_...pkl
└── tsne_output/
    └── bottom_playing_and_alone_tsne_seq_240_latent_64_bottleneck_32_no_velocity_test_depth/
        ├── tsne_results_...csv
        └── cluster_plot_...png
```

## Summary

**Before:** Scripts had hardcoded paths and parameters scattered everywhere 🔴

**Now:** Everything flows from config.yaml → pipeline → scripts ✅

Just edit `config.yaml` and run `python run_pipeline_from_config.py`!

