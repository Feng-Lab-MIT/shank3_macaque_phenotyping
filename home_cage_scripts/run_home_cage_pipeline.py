#!/usr/bin/env python3
"""
Home cage analysis pipeline (cohort 1).

Runs steps in order; use --steps to run only selected steps.
Set thresholds in config.py, or choose recommended values when the pipeline prompts you (see README).
"""
import argparse
import json
import os
from collections import defaultdict

import config
from self_biting import select_frames
from utils import consecutive_chunks, merge_chunks, padding_chunks, convert_numpy_types
from preprocessing import run_median_mean_for_file_list
from big_motion import select_repetitive_motion_frames
from pacing import select_pacing_frames


def step_self_biting(file_names, different_starting=False):
    """Run self-biting detection. Thresholds from config (self_biting_d_threshold, self_biting_v_threshold)."""
    file_names = sorted(file_names)
    columns = ['Day' + str(d) for d in config.PROCESS_DAYS]
    sb = __import__("pandas").DataFrame(index=config.FULL_MONKEY_LIST, columns=columns)
    for file in file_names:
        monkey, monkey_day, frame_list, df = select_frames(
            file,
            d_threshold=config.self_biting_d_threshold,
            v_threshold=config.self_biting_v_threshold,
            visualization=True,
            different_starting=different_starting,
        )
        chunks = consecutive_chunks(frame_list)
        long_chunks = sum(1 for ch in chunks if len(ch) >= 25)
        print(f"Self-biting bouts (≥1 s): {long_chunks}  (total {len(chunks)} bouts, {len(frame_list)} frames)")
        try:
            day_num = int(monkey_day.split('_')[1].replace('day', '')) if '_' in monkey_day else 1
        except (IndexError, ValueError):
            day_num = 1
        if day_num in config.PROCESS_DAYS and monkey in sb.index:
            sb.loc[monkey, 'Day' + str(day_num)] = len(frame_list) / 180000
    return sb


def step_median_mean(file_names):
    """Compute median/mean per frame from monkey_day CSVs; used by big motion and pacing."""
    run_median_mean_for_file_list(file_names, data_folder=config.data_folder, skip_existing=False)
    median_file_names = [
        os.path.join(config.data_folder, f)
        for f in os.listdir(config.data_folder)
        if f.endswith('median_mean.csv')
    ]
    return median_file_names


def step_big_motion(median_file_names, extract_videos=False, different_starting=False):
    """Detect repetitive-motion chunks (select_repetitive_motion_frames); optionally extract video clips."""
    bm_chunk_dic = defaultdict(list)
    for file_name in median_file_names:
        data_file = file_name.rsplit('/', 1)[-1].rsplit('\\', 1)[-1]
        monkey_day = data_file.rsplit('_median', 1)[0]
        bm_chunk_dic[monkey_day] = []
    output_folder = os.path.join(config.video_chunk_folder, 'repetitive_motion') if extract_videos else ""
    select_repetitive_motion_frames(
        median_file_names,
        config.cutoff_freq,
        config.fs,
        config.order,
        config.repetitive_motion_threshold,
        bm_chunk_dic,
        config.raw_video_folder,
        output_folder,
        extract_videos=extract_videos,
        upper_line=config.upper_line,
        different_starting=different_starting,
    )
    return bm_chunk_dic


def step_pacing(median_file_names, different_starting=False):
    """Detect pacing and build pacing_dic. Thresholds from config."""
    pacing_dic = {}
    select_pacing_frames(
        median_file_names,
        config.pacing_threshold,
        config.pacing_v_threshold,
        config.lookahead_start,
        config.lookahead,
        pacing_dic,
        upper_line=config.upper_line,
        pacing_folder=config.pacing_folder,
        different_starting=different_starting,
    )
    return pacing_dic


def main():
    parser = argparse.ArgumentParser(description="Home cage analysis pipeline (cohort 1)")
    parser.add_argument(
        "--steps",
        nargs="+",
        default=config.DEFAULT_STEPS,
        choices=["self_biting", "median_mean", "big_motion", "pacing"],
        help="Steps to run (default: config.DEFAULT_STEPS)",
    )
    parser.add_argument(
        "--video-folders",
        nargs="+",
        default=None,
        help="Directories to search for monkey_day.csv (default: config.DEFAULT_VIDEO_FOLDERS)",
    )
    parser.add_argument(
        "--different-starting",
        action="store_true",
        default=config.USE_DIFFERENT_STARTING_FRAME,
        help="Trim to 2 hours when data is longer (get_starting_frame); no cut if already ≤2 hr",
    )
    parser.add_argument(
        "--no-different-starting",
        dest="different_starting",
        action="store_false",
        help="Use fixed start frame 50 instead of length-based trim",
    )
    parser.add_argument("--extract-videos", action="store_true", help="Extract repetitive-motion video chunks")
    args = parser.parse_args()

    # Check if thresholds are set for the requested steps
    if not config.thresholds_set_for_steps(args.steps):
        missing = config._missing_thresholds_for_steps(args.steps)
        print("The following thresholds are not set in config.py:", ", ".join(missing))
        print("\n  1) Use recommended thresholds for this run")
        print("  2) I will set them in config.py myself (exit and modify config)")
        try:
            choice = input("\nChoice [1/2]: ").strip().lower() or "1"
        except EOFError:
            choice = "2"
        if choice in ("2", "n", "no"):
            print("\nPlease modify config.py (see README for recommended ranges) before running.")
            return
        config.apply_recommended_thresholds(args.steps)
        print("Using recommended thresholds for this run.\n")

    config.ensure_output_directories()

    video_folders = args.video_folders or config.DEFAULT_VIDEO_FOLDERS
    file_names = config.get_file_names(video_folders)
    if not file_names:
        print("No DLC CSV files (monkey_day.csv, e.g. 57_day1.csv) found in", video_folders)
        return

    print("Found", len(file_names), "CSV files.")

    median_file_names = []
    bm_chunk_dic = {}
    pacing_dic = {}

    if "self_biting" in args.steps:
        print("\n--- Step: self_biting ---")
        step_self_biting(file_names, different_starting=args.different_starting)

    if "median_mean" in args.steps:
        print("\n--- Step: median_mean ---")
        median_file_names = step_median_mean(file_names)
        if not median_file_names:
            data_folder = config.data_folder
            if os.path.isdir(data_folder):
                median_file_names = [
                    os.path.join(data_folder, f)
                    for f in os.listdir(data_folder)
                    if f.endswith('median_mean.csv')
                ]
        print("Median/mean files:", len(median_file_names))
    else:
        data_folder = config.data_folder
        if os.path.isdir(data_folder):
            median_file_names = [
                os.path.join(data_folder, f)
                for f in os.listdir(data_folder)
                if f.endswith('median_mean.csv')
            ]

    if "big_motion" in args.steps and median_file_names:
        print("\n--- Step: big_motion ---")
        bm_chunk_dic = step_big_motion(
            median_file_names,
            extract_videos=args.extract_videos,
            different_starting=args.different_starting,
        )
        out_path = os.path.join(config.data_folder, "big_motion_chunks.json")
        os.makedirs(config.data_folder, exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(convert_numpy_types(dict(bm_chunk_dic)), f)
        print("Saved", out_path)
        import pandas as pd
        rows = []
        for monkey_day in sorted(bm_chunk_dic.keys()):
            chunks_list = bm_chunk_dic[monkey_day][0] if bm_chunk_dic[monkey_day] else []
            n_chunks = len(chunks_list)
            total_frames = sum(len(c) for c in chunks_list)
            total_sec = total_frames / 25.0
            rows.append({"monkey_day": monkey_day, "n_chunks": n_chunks, "total_frames": total_frames, "total_seconds": round(total_sec, 1)})
        summary_df = pd.DataFrame(rows)
        print("\nBig motion chunks summary:")
        print(summary_df.to_string(index=False))

    if "pacing" in args.steps and median_file_names:
        print("\n--- Step: pacing ---")
        pacing_dic = step_pacing(median_file_names, different_starting=args.different_starting)
        out_path = os.path.join(config.pacing_folder, "pacing_dic_dlc.json")
        os.makedirs(config.pacing_folder, exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump({k: v for k, v in pacing_dic.items()}, f)
        print("Saved", out_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
