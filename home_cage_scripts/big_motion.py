"""
Repetitive motion detection: high-pass filtered position and thresholds.
Uses median/mean CSV (mean_x, mean_y per frame). select_repetitive_motion_frames
detects frames with high deviation from mean in filtered x/y (upper-cage filter via upper_line).
"""
import os
import traceback
import numpy as np
import pandas as pd
import cv2
from scipy.signal import filtfilt
from scipy.fftpack import fft
from tqdm import tqdm

import config
from utils import (
    butter_highpass,
    consecutive_chunks,
    get_starting_frame,
    merge_chunks,
    padding_chunks,
)


def _extract_chunks_to_video(raw_video_folder, monkey, monkey_day, chunks, output_folder):
    """Write each chunk to an MP4 file."""
    os.makedirs(output_folder, exist_ok=True)
    video_path = os.path.join(raw_video_folder, monkey, f'{monkey_day}_cut_cropped.mp4')
    if not os.path.isfile(video_path):
        video_path = os.path.join(raw_video_folder, monkey, f'{monkey_day}.mp4')
    if not os.path.isfile(video_path):
        print("Video not found:", video_path)
        return
    cap = cv2.VideoCapture(video_path)
    for num_chunk, chunk in enumerate(tqdm(chunks, desc="Outputting video chunks"), start=1):
        start_frame = chunk[0]
        end_frame = chunk[-1]
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        out_path = os.path.join(output_folder, f'{monkey_day}_{num_chunk}.mp4')
        out_fps = cap.get(cv2.CAP_PROP_FPS)
        out_codec = cv2.VideoWriter_fourcc(*'mp4v')
        out_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        out = cv2.VideoWriter(out_path, out_codec, out_fps, out_size)
        for i in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                break
            try:
                out.write(frame)
            except Exception as e:
                print("Error writing frame {}: {}".format(i, e))
                traceback.print_exc()
                break
        cap.release()
        out.release()


def detect_repetitive_motion(x_filt, y_filt, fs, threshold):
    """FFT-based detection of significant frequencies (repetitive motion)."""
    n = len(x_filt)
    freq = np.fft.fftfreq(n, d=1 / fs)
    x_fft = np.abs(fft(x_filt))
    y_fft = np.abs(fft(y_filt))
    significant_x_freq = freq[np.where(x_fft > threshold)[0]]
    significant_y_freq = freq[np.where(y_fft > threshold)[0]]
    return significant_x_freq, significant_y_freq


def select_repetitive_motion_frames(
    median_file_names,
    cutoff_freq,
    fs,
    order,
    threshold,
    bm_chunk_dic,
    video_root_folder,
    output_folder,
    extract_videos=False,
    upper_line=None,
    different_starting=False,
):
    """
    Select frames with repetitive (high-frequency) motion; optionally extract videos.
    bm_chunk_dic is updated in place: key = monkey_day, value = list of chunk lists.
    upper_line: mean_x below this = upper cage (set in config; see README for range).
    """
    import glob
    if upper_line is None:
        upper_line = getattr(config, 'upper_line', None)
    if upper_line is None:
        raise ValueError("Set config.upper_line (see README for recommended range).")
    for file_name in median_file_names:
        data_file = file_name.rsplit('/', 1)[-1].rsplit('\\', 1)[-1]
        monkey_day = data_file.rsplit('_median', 1)[0]
        monkey = monkey_day.rsplit('_', 1)[0]
        day = monkey_day.rsplit('_', 1)[1]
        print(monkey_day, monkey, day)
        df = pd.read_csv(file_name, header=[0], index_col=[0])
        if len(df) < 180000:
            df.index = range(len(df))
        starting_frame = get_starting_frame(monkey_day, df) if different_starting else 50
        if different_starting:
            df = df.loc[starting_frame : 180000 + starting_frame, :]
        else:
            df = df.loc[starting_frame:, :]
        df['upper'] = 0
        df.loc[(df['mean_x'] < upper_line), 'upper'] = 1
        x = df['mean_x'].values.astype(float).flatten()
        y = df['mean_y'].values.astype(float).flatten()
        x = np.nan_to_num(x)
        y = np.nan_to_num(y)
        b, a = butter_highpass(cutoff_freq, fs, order=order)
        x_filt = filtfilt(b, a, x)
        y_filt = filtfilt(b, a, y)
        detect_repetitive_motion(x_filt, y_filt, fs, threshold)
        big_motion_frames = np.where(
            (np.abs(x_filt - np.mean(x_filt)) > threshold)
            | (np.abs(y_filt - np.mean(y_filt)) > threshold)
        )[0]
        big_motion_frames = big_motion_frames + starting_frame
        filtered_big_motion_frames = [f for f in big_motion_frames if df.loc[f, 'upper'] == 1]
        if len(filtered_big_motion_frames) == 0:
            continue
        # 2s padding (front=50, back=50)
        chunks = merge_chunks(padding_chunks(consecutive_chunks(filtered_big_motion_frames), front=50, back=50))
        bm_chunk_dic[monkey_day].append(chunks)
        print(len(chunks), "chunks (after padding and merging)")
        total_big_motion_time = sum(len(c) for c in chunks) / 25
        print("Total big motion time = ", total_big_motion_time, " seconds")
        if extract_videos:
            files = glob.glob1(output_folder, monkey_day + '*.mp4')
            if len(files) == len(chunks):
                continue
            _extract_chunks_to_video(
                video_root_folder, monkey, monkey_day, chunks, output_folder,
            )
