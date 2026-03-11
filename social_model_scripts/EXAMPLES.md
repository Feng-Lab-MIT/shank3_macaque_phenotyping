# Pipeline Configuration Examples

This document provides example configurations for common use cases.

## Example 1: Training a Small Model (Fast)

Good for testing or quick experiments.

```yaml
# config.yaml
model:
  sequence_length: 120        # Shorter sequences (5 seconds)
  prediction_length: 12       # Shorter predictions (0.5 seconds)
  latent_dim: 32             # Smaller model
  bottleneck_dim: 16         # Smaller bottleneck

training:
  batch_size: 64             # Larger batches for faster training
  patience: 20               # Less patience for faster results

pipeline:
  run_training: true
  run_latent_extraction: true
  run_pca: true
  run_tsne: true
```

## Example 2: Training a Large Model (Best Performance)

For best accuracy when you have time and computational resources.

```yaml
# config.yaml
model:
  sequence_length: 480        # Long sequences (20 seconds)
  prediction_length: 48       # Longer predictions (2 seconds)
  latent_dim: 128            # Larger model
  bottleneck_dim: 64         # Larger bottleneck

training:
  batch_size: 16             # Smaller batches (may be necessary for GPU memory)
  patience: 50               # More patience for better convergence

pipeline:
  run_training: true
  run_latent_extraction: true
  run_pca: true
  run_tsne: true
```

## Example 3: Re-running Analysis Only

You already have a trained model and just want to re-run the analysis with different parameters.

```yaml
# config.yaml
model:
  sequence_length: 240        # Must match your existing model!
  prediction_length: 24       # Must match your existing model!
  latent_dim: 64             # Must match your existing model!
  bottleneck_dim: 32         # Must match your existing model!

tsne:
  n_clusters: 64             # More clusters for finer granularity
  perplexity: 100            # Higher perplexity for global structure

pipeline:
  run_training: false         # Skip training - use existing model
  run_latent_extraction: true
  run_pca: true
  run_tsne: true
```

## Example 4: Only Re-running Clustering

You have latent representations and PCA results, just want to try different clustering parameters.

```yaml
# config.yaml
model:
  sequence_length: 240        # Must match your existing files
  latent_dim: 64             # Must match your existing files
  bottleneck_dim: 32         # Must match your existing files

tsne:
  n_clusters: 16             # Fewer clusters for broader categories
  n_neighbors: 100           # More neighbors for smoother clusters

pipeline:
  run_training: false
  run_latent_extraction: false
  run_pca: false
  run_tsne: true             # Only run t-SNE
```

## Example 5: Different Behavioral Categories

Analyzing different behavioral categories.

```yaml
# config.yaml
categories:
  desired_categories:
    - "Top Playing"
    - "Top Alone"

# Note: You'll need to retrain the model for different categories
pipeline:
  run_training: true
  run_latent_extraction: true
  run_pca: true
  run_tsne: true
```

## Example 6: Memory-Constrained System

If you're running on a system with limited GPU memory.

```yaml
# config.yaml
model:
  sequence_length: 120        # Shorter sequences use less memory
  latent_dim: 32             # Smaller model
  bottleneck_dim: 16         # Smaller bottleneck

training:
  batch_size: 8              # Small batch size for limited GPU memory

tsne:
  n_batches: 12              # More batches = less memory usage per batch

pipeline:
  run_training: true
  run_latent_extraction: true
  run_pca: true
  run_tsne: true
```

## Example 7: High-Resolution Analysis

For very detailed temporal analysis.

```yaml
# config.yaml
model:
  sequence_length: 960        # Very long sequences (40 seconds)
  prediction_length: 96       # Long predictions (4 seconds)
  latent_dim: 256            # Very large model
  bottleneck_dim: 128        # Large bottleneck

training:
  batch_size: 4              # Very small batches (necessary for large sequences)
  patience: 100              # Much more patience

pca:
  target_variance: 0.995     # Capture more variance

tsne:
  n_clusters: 128            # Many clusters for fine-grained analysis
  perplexity: 200            # High perplexity

pipeline:
  run_training: true
  run_latent_extraction: true
  run_pca: true
  run_tsne: true
```

## Tips

### Matching Existing Models

**Important:** When using an existing trained model, you MUST use the same hyperparameters:
- `sequence_length`
- `prediction_length`
- `latent_dim`
- `bottleneck_dim`

These are encoded in the model's architecture and cannot be changed without retraining.

### GPU Memory Issues

If you encounter out-of-memory errors:
1. Reduce `batch_size` (e.g., 32 → 16 → 8)
2. Reduce `sequence_length` (e.g., 240 → 120)
3. Reduce `latent_dim` and `bottleneck_dim` (e.g., 64/32 → 32/16)

### Training Time

Approximate training times on a single GPU:
- Small model (32/16, seq=120): ~2-4 hours
- Medium model (64/32, seq=240): ~4-8 hours
- Large model (128/64, seq=480): ~8-16 hours
- Very large model (256/128, seq=960): ~16-32 hours

### PCA Components

The number of PCA components is automatically determined to capture the target variance (default: 99%).
If you want to capture more fine-grained structure, increase `pca.target_variance` to 0.995 or 0.999.

### t-SNE Parameters

- **Perplexity**: Controls local vs global structure
  - Low (15-30): Emphasizes local structure, many small clusters
  - Medium (30-100): Balanced
  - High (100-500): Emphasizes global structure, fewer large clusters

- **n_clusters**: Number of behavioral clusters to identify
  - Fewer clusters (8-16): Broad behavioral categories
  - Medium clusters (16-64): Balanced granularity
  - Many clusters (64-128): Very fine-grained sub-behaviors

