# Home Cage Analysis

Analysis scripts for DeepLabCut (DLC) pose data from macaque home cage videos. Detects and quantifies stereotypic behaviors: self-biting, repetitive (big) motion, and pacing.

## Example data

One example DLC CSV is included: **`data/57_day2.csv`**. You can run the full pipeline with defaults (no path changes needed):

```bash
python run_home_cage_pipeline.py
```

When prompted for thresholds, choose option 1 to use recommended values. Outputs (median/mean CSVs, big motion chunks, pacing dictionary) are written to `./data/`, `./plots/`, `./pacing/`, etc. For your own data, put `monkey_day.csv` files in `./data/` or set **DEFAULT_VIDEO_FOLDERS** in `config.py` / use `--video-folders`.

## Requirements

- Python 3.7+
- DLC CSV outputs (multi-level headers, body part coordinates)

```bash
pip install -r requirements.txt
```

## Configuration

Edit **`config.py`** (and/or set env vars for paths):

- **FULL_MONKEY_LIST** — Monkey IDs for file discovery.
- **DEFAULT_VIDEO_FOLDERS** — Directories containing `monkey_day.csv` (e.g. `57_day2.csv`). Default is `['./data/']` (example file included); set to your path or pass `--video-folders`.
- **PROCESS_DAYS** — Which day numbers to look for (e.g. `[2]` for `*_day2.csv`). Default `[2]` matches the example `57_day2.csv`.
- **Folder paths** — Optional env vars: `HOME_CAGE_PLOT_FOLDER`, `HOME_CAGE_STEREOTYPIC_FOLDER`, `HOME_CAGE_VIDEO_CHUNK_FOLDER`, `HOME_CAGE_RAW_VIDEO_FOLDER`, `HOME_CAGE_DATA_FOLDER`, `HOME_CAGE_PACING_FOLDER`.

## Thresholds

Set these in **`config.py`** for your setup (see table below). If any are unset when you run the pipeline, it will **prompt** you: use recommended thresholds for this run, or exit and set them in config. No error is raised.

| Parameter | Used by | Recommended range | Description |
|-----------|---------|-------------------|-------------|
| `cutoff_freq` | big_motion | 3–6 (Hz) | High-pass filter cutoff |
| `fs` | big_motion | 25 | Video frame rate (fps) |
| `order` | big_motion | 4–6 | Filter order |
| `upper_line` | big_motion, pacing | 300–700 | mean_x below this = upper cage (pixels) |
| `repetitive_motion_threshold` | big_motion | 5–25 | Min deviation from mean in filtered x/y |
| `pacing_threshold` | pacing | 8–15 | Position return threshold (pixels) |
| `pacing_v_threshold` | pacing | 10–20 | Velocity threshold |
| `lookahead` | pacing | 150–250 (frames) | Lookahead window |
| `lookahead_start` | pacing | 40–60 (frames) | Start of lookahead |
| `self_biting_d_threshold` | self_biting | 30–50 | Max distance (pixels) for eye–hand/body |
| `self_biting_v_threshold` | self_biting | 3–8 | Max eye–neck velocity |

## Main steps

1. **File discovery** — `monkey_day.csv` in `DEFAULT_VIDEO_FOLDERS` for each monkey/day in `PROCESS_DAYS`.
2. **Self-biting** — `select_frames()`: distance and velocity criteria; writes selected frames.
3. **Median/mean** — Per-frame median/mean from body parts (for big_motion and pacing).
4. **Big motion** — `select_repetitive_motion_frames()`: high-pass filter + threshold; upper-cage filter; 2s padding.
5. **Pacing** — `select_pacing_frames()`: velocity and return-to-position.

Run: `python run_home_cage_pipeline.py` (or `--steps self_biting median_mean big_motion pacing`).

## Data assumptions

- DLC CSVs: `monkey_day.csv` (e.g. `57_day2.csv`), header `[0,1]`, cohort 1 body-part names. Tail parts (30-mid tail, 31-tail tip) excluded from median/mean; last 3 columns (timestamps) dropped.
- 25 fps. **Length and trimming:** Analyses are designed for up to 2 hours of data (180000 frames). When you use the default **`--different-starting`** behavior, the pipeline calls `get_starting_frame()` (in `utils.py`): if your recording is *longer* than 2 hours, it trims to exactly 2 hours from a small offset so segment length is consistent; if your data is already 2 hours or shorter, **no cut is applied** and all frames from the start are used.

