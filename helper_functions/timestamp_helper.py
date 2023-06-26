"""Creating helper functions to manipulate the Unix Timestamps

Author : Eric Rios and Aditya John"""


def timestamp_corrector(gaze_csv_path, col_name="timestamp [ns]_for_grouping"):
    """Process the unix timestamps
    and create seconds columns to facilitate
    generation of descriptive statistics"""

    gaze_copy = pd.read_csv(gaze_csv_path)
    gaze_copy["ts"] = gaze_copy[col_name].apply(
        lambda x: dt.datetime.fromtimestamp(x / 1000000000)
    )
    baseline = gaze_copy["ts"][0]
    gaze_copy["increment_marker"] = gaze_copy["ts"] - baseline
    gaze_copy["seconds_id"] = gaze_copy["increment_marker"].apply(lambda x: x.seconds)
    return gaze_copy
