"""This script describe statistics of when/where people are looking"""

import pandas as pd
import datetime as dt
from auxiliary_analysis_functions import fake_tagger

# Records most frequently looked at/returned to feature
gaze_copy = pd.read_csv("all_gaze.csv", compression="gzip")


gaze_copy = fake_tagger(gaze_copy)
mode = gaze_copy["tag"].mode()
print(mode)

# Records percent time spent on each feature
gaze_copy["ts"] = gaze_copy["timestamp [ns]"].apply(
    lambda x: dt.datetime.fromtimestamp(x / 1000000000)
)
baseline = gaze_copy["ts"][0]
gaze_copy["increment_marker"] = gaze_copy["ts"] - baseline
gaze_copy["seconds_id"] = gaze_copy["increment_marker"].apply(lambda x: x.seconds)

gaze_copy["next time"] = gaze_copy["ts"].shift(-1)

# Drop last row to get rid of null value since one fixation is short enough that it shouldn't have a major impact on analysis
gaze_nonull = gaze_copy.drop([12760])
gaze_nonull["duration"] = gaze_nonull["next time"] - gaze_nonull["ts"]
gaze_nonull["duration(micro)"] = gaze_nonull["duration"].apply(lambda x: x.microseconds)
gaze_nonull["duration(s)"] = gaze_nonull["duration(micro)"] / 1000000
feature_time = gaze_nonull.groupby("tag")["duration(s)"].sum()
total = gaze_copy["increment_marker"].iloc[-1]
total_time = total.seconds + (total.microseconds / 1000000)
percent_time = (feature_time / total_time) * 100
