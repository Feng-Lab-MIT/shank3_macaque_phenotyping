"""
Home cage analysis for DLC (DeepLabCut) pose data — cohort 1.

This script has been split into modules. Use run_home_cage_pipeline.py to run
the full pipeline. Configure paths and thresholds in config.py (see README).
"""
import config
from utils import (
    filter_consecutive_integers,
    get_starting_frame,
    consecutive_chunks,
    padding_chunks,
    merge_lists,
    merge_chunks,
    butter_highpass,
    convert_numpy_types,
)
from self_biting import select_frames
from big_motion import select_repetitive_motion_frames, detect_repetitive_motion
from pacing import select_pacing_frames, load_pacing_dic
from preprocessing import compute_median_mean_cohort1, run_median_mean_for_file_list

FULL_MONKEY_LIST = config.FULL_MONKEY_LIST
plot_folder = config.plot_folder
stereotypic_folder = config.stereotypic_folder
video_chunk_folder = config.video_chunk_folder
raw_video_folder = config.raw_video_folder
data_folder = config.data_folder
pacing_folder = config.pacing_folder

__all__ = [
    "config",
    "select_frames",
    "filter_consecutive_integers",
    "get_starting_frame",
    "consecutive_chunks",
    "padding_chunks",
    "merge_lists",
    "merge_chunks",
    "butter_highpass",
    "convert_numpy_types",
    "select_repetitive_motion_frames",
    "detect_repetitive_motion",
    "select_pacing_frames",
    "load_pacing_dic",
    "compute_median_mean_cohort1",
    "run_median_mean_for_file_list",
    "FULL_MONKEY_LIST",
]


def _run_pipeline():
    from run_home_cage_pipeline import main
    main()


if __name__ == "__main__":
    _run_pipeline()
