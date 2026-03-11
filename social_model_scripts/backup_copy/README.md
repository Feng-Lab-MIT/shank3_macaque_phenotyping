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

**Option 2: Using the standalone script**
```bash
# Edit hyperparameters in run_pipeline.py
python run_pipeline.py
```

### Customizing Hyperparameters

**Using config.yaml (Recommended):**

Edit the `config.yaml` file:

```yaml
model:
  sequence_length: 240      # Input sequence length (frames)
  prediction_length: 24     # Prediction length (frames)
  latent_dim: 64           # LSTM latent dimension
  bottleneck_dim: 32       # Bottleneck layer dimension
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

```yaml
pipeline:
  run_training: false              # Skip training
  run_latent_extraction: true      # Extract latents from existing model
  run_pca: true                    # Run PCA
  run_tsne: true                   # Run t-SNE
```

**Using run_pipeline.py:**

```python
# Pipeline control - set to False to skip a step
RUN_TRAINING = False              # Skip training
RUN_LATENT_EXTRACTION = True      # Extract latents from existing model
RUN_PCA = True                    # Run PCA
RUN_TSNE = True                   # Run t-SNE
```

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
- `DESIRED_CATEGORIES` - Behavioral categories to analyze (default: ['Bottom Playing', 'Bottom Alone'])
- `PCA_TARGET_VARIANCE` - Target variance to explain with PCA (default: 0.99)
- `TSNE_N_CLUSTERS` - Number of clusters for t-SNE (default: 32)
- `TSNE_PERPLEXITY` - t-SNE perplexity parameter (default: 50)

## Output Structure

The pipeline creates the following output directories:

```
model_2_sub_train_with_sub/
├── best_model_*.h5                          # Trained model
├── feature_info_*.pkl                       # Feature metadata
├── training_log_*.csv                       # Training history
└── feature_distributions/                   # Feature distribution plots

/media/fenglab/newssd/social/
├── bottom_playing_and_alone_latent_*/       # Latent representations
│   ├── latent_representations_*.npy
│   └── indices_*.npy
├── bottom_playing_and_alone_pca_*/          # PCA results
│   └── pca_results/
│       ├── pca_vectors_*.npy
│       ├── pca_model_*.pkl
│       └── frame_indices_*.npy
└── bottom_playing_and_alone_pca_*/          # t-SNE results
    ├── tsne_results_*.csv
    └── cluster_plot_*.png
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

```python
# In run_pipeline.py:
SEQUENCE_LENGTH = 480      # Longer sequences
LATENT_DIM = 128          # Larger model
BOTTLENECK_DIM = 64       # Larger bottleneck

RUN_TRAINING = True
RUN_LATENT_EXTRACTION = True
RUN_PCA = True
RUN_TSNE = True
```

### Case 2: Re-running analysis on existing model

```python
# In run_pipeline.py:
RUN_TRAINING = False              # Skip training
RUN_LATENT_EXTRACTION = True      # Re-extract latents
RUN_PCA = True                    # Re-run PCA
RUN_TSNE = True                   # Re-run t-SNE
```

### Case 3: Only re-running clustering with different parameters

```python
# In run_pipeline.py:
TSNE_N_CLUSTERS = 64              # More clusters

RUN_TRAINING = False
RUN_LATENT_EXTRACTION = False
RUN_PCA = False
RUN_TSNE = True                   # Only run t-SNE
```

## Troubleshooting

### Issue: "Model not found"
- Make sure `RUN_TRAINING = True` or that you have a trained model at the expected path
- Check that `SEQUENCE_LENGTH`, `LATENT_DIM`, and `BOTTLENECK_DIM` match your existing model

### Issue: "Latent directory not found"
- Make sure you ran latent extraction before PCA/t-SNE
- Check that the output paths are correct

### Issue: Out of memory
- Reduce `BATCH_SIZE`
- For PCA/t-SNE on large datasets, the scripts use incremental/batch processing automatically

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

