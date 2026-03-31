"""
Configuration for home cage analysis (cohort 1).
Set paths and parameters here or via environment variables.

THRESHOLDS: Set in config for your setup, or choose "recommended" when the pipeline prompts you.
See README for recommended ranges.
"""
import os

FULL_MONKEY_LIST = [
    '52', '54', '55', '56', '57', '78', '79', '80', '81', '82', '83', '84', '85', '86', '87'
]

# Folder paths — override via env vars for your environment
plot_folder = os.environ.get("HOME_CAGE_PLOT_FOLDER", "./plots/")
stereotypic_folder = os.environ.get("HOME_CAGE_STEREOTYPIC_FOLDER", "./stereotypic/")
video_chunk_folder = os.environ.get("HOME_CAGE_VIDEO_CHUNK_FOLDER", "./video_chunks/")
raw_video_folder = os.environ.get("HOME_CAGE_RAW_VIDEO_FOLDER", "./raw_videos/")
data_folder = os.environ.get("HOME_CAGE_DATA_FOLDER", "./data/")
pacing_folder = os.environ.get("HOME_CAGE_PACING_FOLDER", "./pacing/")

# Default CSV search path — directory containing monkey_day.csv (e.g. 57_day2.csv).
# Example data: one file is provided in data/57_day2.csv so the pipeline runs out of the box.
DEFAULT_VIDEO_FOLDERS = ['./data/']

# Which days to process (1 = day1 only, [2] = day2 only, [1, 2, 3] = all three days).
# Set to [2] to run with the included example data/57_day2.csv.
PROCESS_DAYS = [2]

# Which pipeline steps to run by default (omit or pass --steps to override).
# Options: "self_biting", "median_mean", "big_motion", "pacing"
DEFAULT_STEPS = ["self_biting", "median_mean", "big_motion", "pacing"]

# When True, use get_starting_frame() to trim to exactly 2 hours when data is longer (see utils.py).
USE_DIFFERENT_STARTING_FRAME = True

# ---------------------------------------------------------------------------
# THRESHOLDS — set these for your setup (see README for recommended ranges).
# ---------------------------------------------------------------------------
# High-pass filter (repetitive/big motion): cutoff_freq (Hz), fs (fps), order
cutoff_freq = None   # recommended: 3–6
fs = None            # recommended: 25 (video fps)
order = None         # recommended: 4–6
# Upper vs lower cage: mean_x below this = upper cage (excluded from some analyses)
upper_line = None    # recommended: 300–700 (pixel x)
# Repetitive motion: deviation from mean in filtered x/y above this = motion frame
repetitive_motion_threshold = None   # recommended: 5–25
# Pacing: position return threshold (pixels) and velocity threshold
pacing_threshold = None      # recommended: 8–15
pacing_v_threshold = None    # recommended: 10–20
lookahead = None             # recommended: 150–250 (frames)
lookahead_start = None       # recommended: 40–60 (frames)
# Self-biting: distance threshold (pixels) and eye–neck velocity threshold
self_biting_d_threshold = None   # recommended: 30–50
self_biting_v_threshold = None   # recommended: 3–8

# Recommended values (used if user chooses "use recommended" when config not set)
RECOMMENDED = {
    "cutoff_freq": 4,
    "fs": 25,
    "order": 5,
    "upper_line": 500,
    "repetitive_motion_threshold": 20,
    "pacing_threshold": 10,
    "pacing_v_threshold": 15,
    "lookahead": 200,
    "lookahead_start": 50,
    "self_biting_d_threshold": 40,
    "self_biting_v_threshold": 5,
}


def _missing_thresholds_for_steps(steps):
    """Return sorted list of config attribute names that are None for the given steps."""
    missing = []
    if "self_biting" in steps:
        if self_biting_d_threshold is None:
            missing.append("self_biting_d_threshold")
        if self_biting_v_threshold is None:
            missing.append("self_biting_v_threshold")
    if "big_motion" in steps:
        if cutoff_freq is None:
            missing.append("cutoff_freq")
        if fs is None:
            missing.append("fs")
        if order is None:
            missing.append("order")
        if upper_line is None:
            missing.append("upper_line")
        if repetitive_motion_threshold is None:
            missing.append("repetitive_motion_threshold")
    if "pacing" in steps:
        if pacing_threshold is None:
            missing.append("pacing_threshold")
        if pacing_v_threshold is None:
            missing.append("pacing_v_threshold")
        if lookahead is None:
            missing.append("lookahead")
        if lookahead_start is None:
            missing.append("lookahead_start")
        if upper_line is None:
            missing.append("upper_line")
    return sorted(set(missing))


def thresholds_set_for_steps(steps):
    """Return True if all thresholds required for the given steps are set in config."""
    return len(_missing_thresholds_for_steps(steps)) == 0


def apply_recommended_thresholds(steps):
    """Set any missing thresholds for the given steps to RECOMMENDED values (in-place on config module)."""
    import sys
    mod = sys.modules[__name__]
    for name in _missing_thresholds_for_steps(steps):
        if name in RECOMMENDED:
            setattr(mod, name, RECOMMENDED[name])


def get_file_names(video_folders=None, days=None):
    """
    Build list of DLC CSV paths (monkey_day.csv, e.g. 57_day1.csv) for each monkey/day.
    """
    if video_folders is None:
        video_folders = DEFAULT_VIDEO_FOLDERS
    if days is None:
        days = PROCESS_DAYS
    file_names = []
    for monkey in FULL_MONKEY_LIST:
        for folder in video_folders:
            if not os.path.isdir(folder):
                continue
            for day_num in days:
                f = f"{monkey}_day{day_num}.csv"
                path = os.path.join(folder, f)
                if os.path.isfile(path):
                    file_names.append(path)
    return sorted(file_names)


def ensure_output_directories():
    """Create all output folders from config if they do not exist."""
    for folder in (
        plot_folder,
        stereotypic_folder,
        video_chunk_folder,
        raw_video_folder,
        data_folder,
        pacing_folder,
    ):
        if folder:
            os.makedirs(os.path.abspath(folder), exist_ok=True)

