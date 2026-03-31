# Pipeline Fixes Applied

## Issues Found and Fixed

### Issue 1: Module-Level Code Execution in tsne.py ❌

**Problem:** The `tsne.py` script had most of its code (lines 145-382) running at the module level, outside of any function. This meant the code would execute immediately when imported, using hardcoded parameter values **before** the pipeline could update them.

**Fix:** ✅
- Wrapped all the main logic in a proper `main()` function
- Moved path construction (`base_folder`, `pca_results_folder`, `latent_dir`) inside `main()` so they're built using updated parameters
- Added `global` declarations where needed
- Added proper `if __name__ == "__main__"` block

### Issue 2: Module-Level Path Construction in pca.py ❌

**Problem:** Similar to tsne.py, `pca.py` was constructing paths at the module level (lines 17-18):
```python
latent_dir = f'/path/to/latent_output/latent_seq_{sequence_length}_...'
output_folder = f'/path/to/pca_output/pca_seq_{sequence_length}_...'
```
This happened before the pipeline could update `sequence_length`, `latent_dim`, or `bottleneck_dim`.

**Fix:** ✅
- Changed module-level path variables to `None`
- Moved path construction to the beginning of `main()` function
- Now paths are constructed using the updated parameter values

### Issue 3: Incorrect Default Parameter Values

**Problem:** The scripts had inconsistent default values:
- `tsne.py`: `latent_dim = 512`, `bottleneck_dim = 256`
- `pca.py`: `latent_dim = 512`, `bottleneck_dim = 256`
- `latent_extraction.py`: `latent_dim = 64`, `bottleneck_dim = 32` ✅
- `model_training.py`: `latent_dim = 64`, `bottleneck_dim = 32` ✅

**Fix:** ✅
- Updated all scripts to use consistent defaults: `latent_dim = 64`, `bottleneck_dim = 32`
- These match the config.yaml defaults

## Summary of Changes

### Files Modified:

1. **`tsne.py`** - Complete refactor
   - Wrapped main logic in `main()` function
   - Moved path construction inside `main()`
   - Fixed indentation throughout
   - Updated default parameters
   - Added `sequence_length` parameter to `plot_tsne_with_labels_subset()`

2. **`pca.py`** - Path construction fix
   - Moved path construction from module level to inside `main()`
   - Updated default parameters

3. **`run_pipeline_from_config.py`** - Simplified
   - Removed unnecessary path construction in PCA step
   - Let the scripts construct their own paths internally

## How It Works Now

### Before (Broken) 🔴

```python
# In tsne.py
sequence_length = 240  # Hardcoded
latent_dim = 512      # Hardcoded
base_folder = f'/path/to/...seq_{sequence_length}...'  # Constructed immediately

# When pipeline runs:
import tsne  # ❌ base_folder already constructed with old values!
tsne.sequence_length = 480  # ❌ Too late, paths already built!
tsne.main()  # ❌ Uses wrong paths
```

### After (Fixed) ✅

```python
# In tsne.py
sequence_length = 240  # Default (can be overridden)
latent_dim = 64      # Default (can be overridden)
base_folder = None   # Not constructed yet

def main():
    global base_folder
    # NOW construct paths using current parameter values
    base_folder = f'/path/to/...seq_{sequence_length}...'  # ✅ Uses updated values!
    # ... rest of code

# When pipeline runs:
import tsne  # ✅ No paths constructed yet
tsne.sequence_length = 480  # ✅ Parameter updated
tsne.main()  # ✅ Paths constructed with correct values!
```

## Testing the Fix

To verify the fix works, try running with custom parameters:

```yaml
# config.yaml
model:
  sequence_length: 120  # Different from default
  latent_dim: 32       # Different from default
```

Then run:
```bash
python run_pipeline_from_config.py
```

The scripts should now correctly use `sequence_length=120` and `latent_dim=32` in all their file paths and processing.

## Files Status

| Script | Main() Function | Paths in Main() | Default Params | Status |
|--------|----------------|-----------------|----------------|--------|
| `model_training.py` | ✅ | ✅ | ✅ (64/32) | Good |
| `latent_extraction.py` | ✅ | ✅ | ✅ (64/32) | Good |
| `pca.py` | ✅ | ✅ Fixed | ✅ Fixed (64/32) | **Fixed** |
| `tsne.py` | ✅ Added | ✅ Fixed | ✅ Fixed (64/32) | **Fixed** |

## Remaining Considerations

1. **Path Conventions**: The scripts still use hardcoded path patterns like:
   - `/path/to/latent_output/bottom_playing_and_alone_latent_seq_...`
   
   These could be made more flexible by reading from config, but that would require more extensive refactoring.

2. **Backwards Compatibility**: The scripts can still be run standalone (not through the pipeline) and will use their default values.

3. **Config.yaml Paths**: The user updated paths in config.yaml to be cleaner:
   ```yaml
   paths:
     model_output: "/path/to/model_output/"
     latent_output: "/path/to/latent_output/"
   ```
   However, the scripts still append their own subdirectory names. This is fine and maintains the existing file organization.

## Next Steps

The pipeline should now work correctly! Try running:

```bash
python run_pipeline_from_config.py
```

All hyperparameters will be properly propagated through the entire pipeline.

