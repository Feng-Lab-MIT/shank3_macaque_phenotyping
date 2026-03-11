# Attention Model Pipeline

This pipeline runs the complete attention model workflow for behavioral analysis.

## Pipeline Steps

1. **Model Training** (`model_training.py`) - Trains the attention-based LSTM model
2. **Latent Extraction** (`latent_extraction.py`) - Extracts latent representations from the trained model
3. **PCA Analysis** (`pca.py`) - Performs dimensionality reduction on latent space
4. **t-SNE Analysis** (`tsne.py`) - Visualizes latent space and performs clustering

## Quick Start

### Running the Complete Pipeline

You can run the pipeline in two ways:

**Option 1: Using the config file (Recommended)**
```bash
# Edit config.yaml to set your hyperparameters
python run_pipeline_from_config.py
```

**Option 2: Running scripts standalone**
```bash
# Edit hyperparameters at the top of each script, then run in order:
python model_training.py
python latent_extraction.py
python pca.py
python tsne.py
```

### Customizing Hyperparameters

**Using config.yaml (Recommended):**

Edit the `config.yaml` file. All paths and hyperparameters are set there:

```yaml
paths:
  data: "/path/to/your/all_dfs.pkl"
  category_file: "/path/to/your/data_for_analysis_..._clusters.csv"
  model_output: "/path/to/model_output/"
  latent_output: "/path/to/latent_output/"
  pca_output: "/path/to/pca_output/"
  tsne_output: "/path/to/tsne_output/"

model:
  sequence_length: 48
  prediction_length: 24
  latent_dim: 64
  bottleneck_dim: 32

categories:
  desired_categories: "all"   # Use "all" for all categories, or list e.g. ["Bottom Playing", "Bottom Alone"]
```

**Using run_pipeline.py:**

Edit the hyperparameters section at the top of `run_pipeline.py`:

```python
# Model architecture parameters
SEQUENCE_LENGTH = 240      # Input sequence length (in frames)
PREDICTION_LENGTH = 24     # Prediction length (in frames)
LATENT_DIM = 64           # LSTM latent dimension
BOTTLENECK_DIM = 32       # Bottleneck layer dimension
```

### Skipping Steps

You can skip individual steps by setting the corresponding flag to `false`/`False`:

**Using config.yaml:**

Set the corresponding flag to `false` in the `pipeline` section:

```yaml
pipeline:
  run_training: false
  run_latent_extraction: true
  run_pca: true
  run_tsne: true
```

**When running scripts standalone:** Edit the pipeline control section in each script if needed.

## Key Hyperparameters

### Model Architecture
- `SEQUENCE_LENGTH` - Length of input sequences (default: 240 frames = 10 seconds at 24fps)
- `PREDICTION_LENGTH` - Length of predictions (default: 24 frames = 1 second)
- `LATENT_DIM` - LSTM hidden dimension (default: 64)
- `BOTTLENECK_DIM` - Bottleneck layer dimension (default: 32)

### Training Parameters
- `BATCH_SIZE` - Batch size for training (default: 32)
- `PATIENCE` - Early stopping patience (default: 30)
- `TEST_SIZE` - Validation split ratio (default: 0.3)
- `DROPOUT_RATE` - Dropout rate (default: 0.2)
- `REGULARIZATION_LAMBDA` - L2 regularization strength (default: 0.1)

### Analysis Parameters
- **Behavioral categories** (`categories.desired_categories` in config): Use `"all"` to include all categories, or a list e.g. `["Bottom Playing", "Bottom Alone"]`
- `target_variance` (PCA): Target variance to explain (default: 0.99)
- `n_clusters`, `perplexity` (t-SNE): Clustering and t-SNE parameters

## Output Structure

The pipeline creates output directories under the paths specified in `config.yaml` (`paths.model_output`, `paths.latent_output`, etc.). Subfolders are named using the category setting (e.g. `all_categories_*` when using all categories, or `bottom_playing_and_alone_*` for a subset). Example:

```
model_output/
├── best_model_*.h5
├── feature_info_*.pkl
├── training_log_*.csv
└── ...

latent_output/
├── all_categories_latent_seq_*_latent_*_bottleneck_*_no_velocity_test_depth/
│   ├── latent_representations_*.npy
│   └── indices_*.npy
...
```

## Running Individual Scripts

If you need to run scripts individually (not recommended):

```bash
# 1. Train model
python model_training.py

# 2. Extract latents
python latent_extraction.py

# 3. Run PCA
python pca.py

# 4. Run t-SNE
python tsne.py
```

**Note:** When running individually, you must manually update hyperparameters in each script to ensure consistency.

## Common Use Cases

### Case 1: Training a new model with different hyperparameters

Edit `config.yaml`: change `model.sequence_length`, `model.latent_dim`, `model.bottleneck_dim`, etc. Then run:

```bash
python run_pipeline_from_config.py
```

### Case 2: Re-running analysis on existing model

```python
# In run_pipeline.py:
RUN_TRAINING = False              # Skip training
RUN_LATENT_EXTRACTION = True      # Re-extract latents
RUN_PCA = True                    # Re-run PCA
RUN_TSNE = True                   # Re-run t-SNE
```

### Case 3: Only re-running t-SNE with different parameters

In `config.yaml`, set `pipeline.run_training`, `run_latent_extraction`, and `run_pca` to `false`, and `run_tsne` to `true`. Adjust `tsne.n_clusters`, `tsne.perplexity`, etc.

## Troubleshooting

### Issue: "Model not found"
- Ensure `pipeline.run_training` is `true` for a fresh run, or that a trained model exists at `paths.model_output` with matching `sequence_length`, `latent_dim`, and `bottleneck_dim` (and `category_folder_name`, e.g. `all_categories` or `bottom_playing_and_alone`).

### Issue: "Latent directory not found"
- Run latent extraction before PCA/t-SNE, and ensure `paths.latent_output` and the category setting in config match the folder names.

### Issue: Out of memory
- Reduce `training.batch_size` in config (or `batch_size` in the script when running standalone).
- PCA/t-SNE use incremental/batch processing; if needed, reduce data or adjust batch settings in the scripts.

## Environment

Make sure you're running in the correct conda environment:

```bash
conda activate tmpenv
```

### Dependencies

If using the config-based approach (`run_pipeline_from_config.py`), you need PyYAML:

```bash
pip install pyyaml
```

All other dependencies are already included in the existing scripts.

## GPU Configuration

The scripts automatically detect and use available GPUs. Model training uses GPU 0, latent extraction uses GPU 1 (if available).

