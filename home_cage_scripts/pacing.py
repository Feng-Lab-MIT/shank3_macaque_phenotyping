"""
Pacing detection: repetitive back-and-forth (velocity + return-to-position).
Requires median_mean CSV (mean_x, mean_y per frame).
"""
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import config
from utils import consecutive_chunks, get_starting_frame


def select_pacing_frames(
    median_file_names,
    threshold,
    v_threshold,
    lookahead_start,
    lookahead,
    pacing_dic,
    upper_line=None,
    pacing_folder=None,
    different_starting=False,
):
    """
    Detect pacing frames: high velocity and return to similar position within lookahead.
    pacing_dic is updated in place: key = monkey_day, value = list of pacing chunks.
    """
    if upper_line is None:
        upper_line = config.upper_line
    if pacing_folder is None:
        pacing_folder = config.pacing_folder
    os.makedirs(pacing_folder, exist_ok=True)

    for file_name in median_file_names:
        data_file = file_name.rsplit('/', 1)[-1].rsplit('\\', 1)[-1]
        monkey_day = data_file.rsplit('_median', 1)[0]
        monkey = monkey_day.rsplit('_', 1)[0]
        day = monkey_day.rsplit('_', 1)[1]
        print(monkey_day, monkey, day)
        output_name = os.path.join(
            pacing_folder,
            f'{monkey_day}_lhs_{lookahead_start}_lh_{lookahead}_v_threshold_{v_threshold}_threshold_{threshold}_exclude_upper'
            + ('_diff_start' if different_starting else '')
            + '_dlc.csv',
        )
        if os.path.exists(output_name):
            print("Already calculated, skipping...")
            df = pd.read_csv(output_name, header=[0], index_col=[0])
            pacing_frames = [i for i in df.index if df.loc[i, 'removed'] == 1]
            pacing_dic[monkey_day] = consecutive_chunks(pacing_frames)
            continue
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
        n_lower = (df['upper'] == 0).sum()
        df['v'] = np.sqrt(
            np.add(
                np.power(df['mean_x'].shift() - df['mean_x'], 2),
                np.power(df['mean_y'].shift() - df['mean_y'], 2),
            )
        )
        df['presum'] = df['v'].cumsum(axis=0, skipna=True).shift()
        df['presum'] = df['presum'].fillna(0)
        df['returned_at'] = 0
        df['removed'] = 0
        df['avg_v'] = 0.0  # float so we can assign cur_avg without dtype warning
        df['pacing_time_length'] = 0
        for i in tqdm(df.index):
            if np.isnan(df.loc[i, 'mean_x']) or np.isnan(df.loc[i, 'mean_y']):
                continue
            if df.loc[i, 'upper'] == 1:
                continue
            start = i + lookahead_start
            end = min(i + lookahead, df.index[-1])
            for j in range(start, end + 1):
                if j + 1 > df.index[-1]:
                    break
                if df.loc[j, 'upper'] == 1:
                    break
                if np.isnan(df.loc[j, 'mean_x']) or np.isnan(df.loc[j, 'mean_y']):
                    continue
                cur_avg = (df.loc[j + 1, 'presum'] - df.loc[i, 'presum']) / (j + 1 - i)
                if cur_avg > v_threshold:
                    if (
                        abs(df.loc[j, 'mean_x'] - df.loc[i, 'mean_x']) < threshold
                        and abs(df.loc[j, 'mean_y'] - df.loc[i, 'mean_y']) < threshold
                    ):
                        df.loc[i, 'removed'] = 1
                        df.loc[i, 'avg_v'] = cur_avg
                        df.loc[i, 'returned_at'] = j
                        df.loc[i, 'pacing_time_length'] = j - i
                        break
        df = df[['mean_x', 'mean_y', 'upper', 'v', 'returned_at', 'removed', 'avg_v', 'pacing_time_length']]
        df.to_csv(output_name)
        pacing_frames = [i for i in df.index if df.loc[i, 'removed'] == 1]
        pacing_chunks = consecutive_chunks(pacing_frames)
        pacing_dic[monkey_day] = pacing_chunks
        prop = sum(len(lst) for lst in pacing_chunks) / len(df) if len(df) else 0
        print(f"Pacing: {n_lower} frames in lower cage (mean_x>={upper_line}), {len(pacing_frames)} pacing frames, proportion={prop:.6f}")


def load_pacing_dic(median_file_names, pacing_folder=None, threshold=10, v_threshold=15, lookahead_start=50, lookahead=200, different_starting=False):
    """
    Load pacing_dic from existing CSV outputs (same naming as select_pacing_frames).
    """
    if pacing_folder is None:
        pacing_folder = config.pacing_folder
    pacing_dic = {}
    suffix = ('_diff_start' if different_starting else '') + '_dlc.csv'
    for file_name in median_file_names:
        data_file = file_name.rsplit('/', 1)[-1].rsplit('\\', 1)[-1]
        monkey_day = data_file.rsplit('_median', 1)[0]
        pacing_file = os.path.join(
            pacing_folder,
            f'{monkey_day}_lhs_{lookahead_start}_lh_{lookahead}_v_threshold_{v_threshold}_threshold_{threshold}_exclude_upper'
            + suffix,
        )
        if not os.path.isfile(pacing_file):
            continue
        df = pd.read_csv(pacing_file, header=[0], index_col=[0])
        pacing_frames = [i for i in df.index if df.loc[i, 'removed'] == 1]
        pacing_dic[monkey_day] = consecutive_chunks(pacing_frames)
    return pacing_dic
