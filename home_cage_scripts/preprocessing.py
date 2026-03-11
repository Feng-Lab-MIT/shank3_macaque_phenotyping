"""
Preprocessing: compute per-frame median/mean position from DLC CSVs.
Outputs median_mean CSV used by big motion and pacing (columns: mean_x, mean_y, median_x, median_y).
Input: monkey_day.csv (e.g. 57_day1.csv) with DLC body-part coordinates; no mean columns.
Skips tail body parts (30-mid tail, 31-tail tip) and drops last 3 columns (timestamps).
"""
import os
import pandas as pd
import numpy as np

import config

# Body parts to exclude from median/mean (tail)
TAIL_BODYPARTS = ('30-mid tail', '31-tail tip')
# Number of trailing columns to drop (timestamps added manually)
NUM_TIMESTAMP_COLUMNS = 3


def _first_level(col):
    """Return first level of a MultiIndex column for matching body part names."""
    if isinstance(col, str):
        return col
    if isinstance(col, (tuple, list)) and len(col):
        return col[0]
    return col


def compute_median_mean_cohort1(csv_path, output_path=None):
    """
    Build median_mean CSV for pacing/big_motion from DLC body-part coordinates.
    - Drops last NUM_TIMESTAMP_COLUMNS columns (timestamps).
    - Excludes body parts in TAIL_BODYPARTS (30-mid tail, 31-tail tip).
    - Computes median_x, median_y, mean_x, mean_y across remaining body-part x/y.
    """
    df = pd.read_csv(csv_path, header=[0, 1], index_col=[0])

    # Remove last 3 columns (timestamps)
    if len(df.columns) >= NUM_TIMESTAMP_COLUMNS:
        df = df.iloc[:, :-NUM_TIMESTAMP_COLUMNS]

    # Get x and y sub-dataframes; exclude tail body parts by header
    df_x = df.xs('x', axis=1, level=1, drop_level=False)
    df_y = df.xs('y', axis=1, level=1, drop_level=False)

    # Drop tail body parts (match first level of column)
    def is_tail(col):
        return _first_level(col) in TAIL_BODYPARTS

    tail_x = [c for c in df_x.columns if is_tail(c)]
    tail_y = [c for c in df_y.columns if is_tail(c)]
    df_x = df_x.drop(columns=tail_x, errors='ignore')
    df_y = df_y.drop(columns=tail_y, errors='ignore')

    out = pd.DataFrame({
        'median_x': df_x.median(axis=1),
        'median_y': df_y.median(axis=1),
        'mean_x': df_x.mean(axis=1),
        'mean_y': df_y.mean(axis=1),
    })

    if output_path is None:
        base = os.path.basename(csv_path).replace('.csv', '')
        output_path = os.path.join(os.path.dirname(csv_path), f'{base}_median_mean.csv')
    out.to_csv(output_path)
    return output_path


def run_median_mean_for_file_list(file_names, data_folder=None, skip_existing=True):
    """
    For each monkey_day CSV in file_names, write median_mean CSV into data_folder.
    data_folder defaults to config.data_folder. If data_folder is None, writes next to each CSV.
    """
    if data_folder is None:
        data_folder = config.data_folder
    os.makedirs(data_folder, exist_ok=True)
    for file_name in file_names:
        base = os.path.basename(file_name).replace('.csv', '')
        out_path = os.path.join(data_folder, f'{base}_median_mean.csv') if data_folder else os.path.join(os.path.dirname(file_name), f'{base}_median_mean.csv')
        if skip_existing and out_path and os.path.exists(out_path):
            continue
        try:
            compute_median_mean_cohort1(file_name, output_path=out_path)
        except Exception as e:
            print("Error processing", file_name, e)
