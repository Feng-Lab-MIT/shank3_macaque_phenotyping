# Quick Start Guide

## TL;DR

To run the complete pipeline with default settings:

```bash
pip install pyyaml  # Only needed once
python run_pipeline_from_config.py
```

That's it! The pipeline will:
1. ✅ Train an attention model
2. ✅ Extract latent representations
3. ✅ Perform PCA dimensionality reduction
4. ✅ Generate t-SNE visualizations with clustering

---

## Customizing Hyperparameters

### Step 1: Edit config.yaml

Open `config.yaml` and change the parameters you want:

```yaml
model:
  sequence_length: 240      # ← Change this
  latent_dim: 64           # ← Or this
```

### Step 2: Run the pipeline

```bash
python run_pipeline_from_config.py
```

---

## Common Scenarios

### Scenario 1: Testing with Faster Settings

Edit `config.yaml`:
```yaml
model:
  sequence_length: 120      # Shorter = faster
  latent_dim: 32           # Smaller = faster

training:
  batch_size: 64           # Larger = faster (if GPU memory allows)
```

### Scenario 2: Using an Existing Model

Edit `config.yaml`:
```yaml
pipeline:
  run_training: false       # Don't retrain
  run_latent_extraction: true
  run_pca: true
  run_tsne: true
```

**Important:** Make sure `model.sequence_length`, `model.latent_dim`, and `model.bottleneck_dim` match your existing model!

### Scenario 3: Only Changing Clustering

Edit `config.yaml`:
```yaml
tsne:
  n_clusters: 64           # Try more clusters

pipeline:
  run_training: false
  run_latent_extraction: false
  run_pca: false
  run_tsne: true           # Only re-run t-SNE
```

---

## File Structure

After creating the pipeline scripts, your directory looks like this:

```
pipeline/
├── model_training.py              # Original scripts (don't edit these)
├── latent_extraction.py
├── pca.py
├── tsne.py
│
├── run_pipeline.py                # Option 1: Standalone pipeline
├── run_pipeline_from_config.py   # Option 2: Config-based pipeline (recommended)
├── config.yaml                    # ← Edit this to change hyperparameters
│
├── README.md                      # Full documentation
├── EXAMPLES.md                    # Example configurations
└── QUICKSTART.md                  # This file
```

---

## What Gets Created

When you run the pipeline, it creates these outputs:

```
model_2_sub_train_with_sub/
├── best_model_*.h5                    # Trained model ✅
├── training_log_*.csv                 # Training history
└── feature_info_*.pkl                 # Feature metadata

/path/to/latent_output/
├── bottom_playing_and_alone_latent_*/  # Latent representations ✅
├── bottom_playing_and_alone_pca_*/     # PCA results ✅
│   └── pca_results/
│       ├── pca_vectors_*.npy
│       └── pca_model_*.pkl
└── bottom_playing_and_alone_pca_*/     # t-SNE results ✅
    ├── tsne_results_*.csv              # Cluster assignments
    └── cluster_plot_*.png              # Visualization
```

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'yaml'"

Install PyYAML:
```bash
pip install pyyaml
```

### "Model not found"

Either:
1. Set `run_training: true` in config.yaml, or
2. Make sure model parameters match your existing model

### "Out of memory"

Reduce these in config.yaml:
```yaml
model:
  sequence_length: 120     # Reduce this
  latent_dim: 32          # Reduce this

training:
  batch_size: 8           # Reduce this
```

### Pipeline is too slow

Speed up training:
```yaml
model:
  sequence_length: 120     # Shorter sequences
  latent_dim: 32          # Smaller model

training:
  batch_size: 64          # Larger batches
  patience: 20            # Less patience
```

---

## Next Steps

1. ✅ Run the default pipeline to make sure everything works
2. 📖 Read `EXAMPLES.md` for common configuration patterns
3. 🔧 Customize `config.yaml` for your specific needs
4. 📚 Check `README.md` for detailed documentation

---

## Getting Help

- See `README.md` for full documentation
- See `EXAMPLES.md` for configuration examples
- Check the individual scripts for implementation details

---

## The Old Way vs The New Way

### Old Way (Manual) ❌
```bash
# Edit model_training.py line 58
python model_training.py

# Edit latent_extraction.py line 23
python latent_extraction.py

# Edit pca.py line 13
python pca.py

# Edit tsne.py line 150
python tsne.py
```

**Problems:**
- Have to edit 4 different files
- Easy to forget to update a parameter
- Time consuming

### New Way (Automated) ✅
```bash
# Edit config.yaml once
python run_pipeline_from_config.py
```

**Benefits:**
- Edit one file
- All parameters stay consistent
- Run everything at once
- Can skip steps easily

---

Happy analyzing! 🎉

