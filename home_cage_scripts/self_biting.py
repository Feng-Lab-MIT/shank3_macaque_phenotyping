"""
Self-biting detection: eye–hand/body distance and eye–neck velocity criteria.
Cohort 1: DLC CSV (monkey_day.csv, e.g. 57_day1.csv) with header=[0,1].
"""
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import config
from utils import filter_consecutive_integers, get_starting_frame


def select_frames(
    file,
    d_threshold=40,
    v_threshold=5,
    visualization=True,
    different_starting=False,
    plot_folder=None,
    stereotypic_folder=None,
):
    """
    Detect self-biting candidate frames from DLC CSV (monkey_day.csv, header=[0,1]).
    Returns (monkey, monkey_day, frame_list, df).
    """
    if plot_folder is None:
        plot_folder = config.plot_folder
    if stereotypic_folder is None:
        stereotypic_folder = config.stereotypic_folder

    data_file = file.rsplit('/', 1)[-1].rsplit('\\', 1)[-1]
    base = data_file.rsplit('.csv', 1)[0]
    monkey_day = base
    parts = monkey_day.split('_')
    monkey = parts[0] if parts else base
    day = parts[1] if len(parts) > 1 else 'day1'
    print(monkey_day, monkey, day)

    df = pd.read_csv(file, header=[0, 1], index_col=[0])
    # Drop last 3 columns (timestamps), same as preprocessing
    if len(df.columns) >= 3:
        df = df.iloc[:, :-3]
    if different_starting:
        starting_frame = get_starting_frame(monkey_day, df)
        df = df.loc[starting_frame:180000 + starting_frame, :]
    else:
        df = df.loc[:180000, :]

    df_y = df.xs('y', axis=1, level=1, drop_level=False)
    median_y = df_y.median(axis=1)
    df_x = df.xs('x', axis=1, level=1, drop_level=False)
    median_x = df_x.median(axis=1)
    df.columns = df.columns.map('_'.join)

    # Cohort 1 body part column names (numbered)
    df['mid_eye_x'] = df[['1-left eye_x', '2-right eye_x']].mean(axis=1)
    df['mid_eye_y'] = df[['1-left eye_y', '2-right eye_y']].mean(axis=1)
    df['en_x'] = df[['mid_eye_x', '6-neck_x']].mean(axis=1)
    df['en_y'] = df[['mid_eye_y', '6-neck_y']].mean(axis=1)
    df['d_mid_eye_left_hand'] = np.sqrt(
        np.add(
            np.power(df['mid_eye_x'] - df['9-left hand_x'], 2),
            np.power(df['mid_eye_y'].shift() - df['9-left hand_y'], 2),
        )
    )
    df['d_mid_eye_right_hand'] = np.sqrt(
        np.add(
            np.power(df['mid_eye_x'] - df['12-right hand_x'], 2),
            np.power(df['mid_eye_y'].shift() - df['12-right hand_y'], 2),
        )
    )
    df['d_mid_eye_left_foot'] = np.sqrt(
        np.add(
            np.power(df['mid_eye_x'] - df['17-left foot_x'], 2),
            np.power(df['mid_eye_y'].shift() - df['17-left foot_y'], 2),
        )
    )
    df['d_mid_eye_right_foot'] = np.sqrt(
        np.add(
            np.power(df['mid_eye_x'] - df['20-right foot_x'], 2),
            np.power(df['mid_eye_y'].shift() - df['20-right foot_y'], 2),
        )
    )
    df['d_mid_eye_left_elbow'] = np.sqrt(
        np.add(
            np.power(df['mid_eye_x'] - df['8-left elbow_x'], 2),
            np.power(df['mid_eye_y'].shift() - df['8-left elbow_y'], 2),
        )
    )
    df['d_mid_eye_right_elbow'] = np.sqrt(
        np.add(
            np.power(df['mid_eye_x'] - df['11-right elbow_x'], 2),
            np.power(df['mid_eye_y'].shift() - df['11-right elbow_y'], 2),
        )
    )
    df['d_mid_eye_left_knee'] = np.sqrt(
        np.add(
            np.power(df['mid_eye_x'] - df['16-left knee_x'], 2),
            np.power(df['mid_eye_y'].shift() - df['16-left knee_y'], 2),
        )
    )
    df['d_mid_eye_right_knee'] = np.sqrt(
        np.add(
            np.power(df['mid_eye_x'] - df['19-right knee_x'], 2),
            np.power(df['mid_eye_y'].shift() - df['19-right knee_y'], 2),
        )
    )
    df['en_v'] = np.sqrt(
        np.add(
            np.power(df['en_x'].shift() - df['en_x'], 2),
            np.power(df['en_y'].shift() - df['en_y'], 2),
        )
    )
    df['depth'] = median_y
    df['x_position'] = median_x
    df['min_d_mid_eye_hands'] = df[['d_mid_eye_left_hand', 'd_mid_eye_right_hand']].min(axis=1)
    df['min_d_mid_eye_feet'] = df[['d_mid_eye_left_foot', 'd_mid_eye_right_foot']].min(axis=1)
    df['min_d_mid_eye_elbows'] = df[['d_mid_eye_left_elbow', 'd_mid_eye_right_elbow']].min(axis=1)
    df['min_d_mid_eye_knees'] = df[['d_mid_eye_left_knee', 'd_mid_eye_right_knee']].min(axis=1)
    df['selected'] = np.nan
    df.loc[
        (
            (df['min_d_mid_eye_hands'] < d_threshold)
            | (df['min_d_mid_eye_feet'] < d_threshold)
            | (df['min_d_mid_eye_elbows'] < d_threshold)
            | (df['min_d_mid_eye_knees'] < d_threshold)
        )
        & (df['en_v'] < v_threshold),
        'selected',
    ] = 1

    frame_list = list(df[df['selected'] == 1].index)
    to_be_added = []
    for i, idx in enumerate(frame_list):
        if i + 1 < len(frame_list) and 1 < frame_list[i + 1] - frame_list[i] < 10:
            for j in range(frame_list[i] + 1, frame_list[i + 1]):
                to_be_added.append(j)
    frame_list.extend(to_be_added)
    frame_list.sort()
    frame_list = filter_consecutive_integers(frame_list)
    selected = df.loc[frame_list]
    frame_list = list(selected.index)
    os.makedirs(os.path.abspath(stereotypic_folder), exist_ok=True)
    selected.to_csv(os.path.join(stereotypic_folder, f'{monkey_day}_selected_v1.csv'))

    if visualization:
        import matplotlib.pyplot as plt
        df['self_biting'] = 0
        for i in tqdm(df.index):
            if i in frame_list:
                df.loc[i, 'self_biting'] = 1
        plt.plot(df['self_biting'])
        plt.title(f'{monkey_day} self biting distribution')
        os.makedirs(os.path.abspath(plot_folder), exist_ok=True)
        plt.savefig(os.path.join(plot_folder, f'{monkey_day}_self_biting_distribution.png'), dpi=300)
        plt.close('all')

    return monkey, monkey_day, frame_list, df
