"""
Shared utilities for home cage analysis: chunk helpers, filtering, type conversion.
"""
import traceback
import numpy as np
from scipy.signal import butter, filtfilt


def filter_consecutive_integers(nums, min_consecutive=10):
    """Keep only runs of consecutive integers with length >= min_consecutive."""
    result = []
    start = 0
    while start < len(nums):
        end = start
        while end + 1 < len(nums) and nums[end + 1] == nums[end] + 1:
            end += 1
        if end - start + 1 >= min_consecutive:
            result.extend(nums[start:end + 1])
        start = end + 1
    return result


def get_starting_frame(monkey_day, df, default_extra_cut=1500):
    """
    Return starting frame so that analysis uses exactly 2 hours of data (180000 frames at 25 fps).

    If the recording is longer than 181500 frames, returns default_extra_cut so that
    [start : start + 180000] yields a 2-hour chunk. If the data is already that length or
    shorter, returns 0 (use from the beginning; no cut). Your data does not need to be
    exactly 2 hours: shorter recordings are used in full; longer ones are trimmed to 2 hours.
    """
    if len(df) > 181500:
        return default_extra_cut
    return 0


def consecutive_chunks(lst):
    """Split a list of integers into lists of consecutive runs."""
    if not lst:
        return []
    result = []
    temp = [lst[0]]
    for i in range(1, len(lst)):
        if lst[i] == lst[i - 1] + 1:
            temp.append(lst[i])
        else:
            result.append(temp)
            temp = [lst[i]]
    result.append(temp)
    return result


def padding_chunks(lst, front=250, back=250):
    """Add padding frames before/after each consecutive chunk."""
    padded_lists = []
    for chunk in lst:
        padded = (
            list(range(max(chunk[0] - front, 0), chunk[0]))
            + chunk
            + list(range(chunk[-1] + 1, chunk[-1] + back + 1))
        )
        padded_lists.append(padded)
    return padded_lists


def merge_lists(list1, list2):
    """Merge two sorted lists and remove duplicates."""
    combined = sorted(list1 + list2)
    deduped = []
    for num in combined:
        if num not in deduped:
            deduped.append(num)
    return deduped


def merge_chunks(lst):
    """Merge overlapping or adjacent chunks into single chunks."""
    if not lst:
        return []
    merged_chunks = []
    cur_chunk = list(lst[0])
    i = 1
    while i < len(lst):
        if lst[i][0] <= cur_chunk[-1]:
            if lst[i][-1] > cur_chunk[-1]:
                cur_chunk = merge_lists(cur_chunk, lst[i])
        else:
            merged_chunks.append(cur_chunk)
            cur_chunk = list(lst[i])
        i += 1
    merged_chunks.append(cur_chunk)
    return merged_chunks


def butter_highpass(cutoff, fs, order=5):
    """Design a high-pass Butterworth filter."""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a


def convert_numpy_types(obj):
    """Convert NumPy types to native Python for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {convert_numpy_types(k): convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(x) for x in obj]
    return obj


# Re-export for convenience
__all__ = [
    'filter_consecutive_integers',
    'get_starting_frame',
    'consecutive_chunks',
    'padding_chunks',
    'merge_lists',
    'merge_chunks',
    'butter_highpass',
    'convert_numpy_types',
]
