"""This script describe statistics of when/where people are looking"""

import pandas as pd
import datetime as dt
import numpy as np
from auxiliary_analysis_functions import fake_tagger


# Groupby function
def modify(df):
    df.reset_index(inplace=True, drop=True)
    baseline = df["ts"][0]
    df["increment_marker"] = df["ts"] - baseline
    df["next time"] = df["ts"].shift(-1)
    df["next x"] = df["gaze x [px]"].shift(-1)
    df["next y"] = df["gaze y [px]"].shift(-1)
    df = df[:-1].copy()
    return df


# Records most frequently looked at/returned to feature
gaze_copy = pd.read_csv("all_gaze.csv", compression="gzip")

gaze_copy = fake_tagger(gaze_copy)
gaze_copy.reset_index(drop=True, inplace=True)
mode = gaze_copy["tag"].mode()
print(mode)

# Records percent time spent on each feature
gaze_copy["ts"] = gaze_copy["timestamp [ns]"].apply(
    lambda x: dt.datetime.fromtimestamp(x / 1000000000)
)
gaze_copy = gaze_copy.groupby(["section id"]).apply(modify)
gaze_copy["seconds_id"] = gaze_copy["increment_marker"].apply(lambda x: x.seconds)

# Drop last row to get rid of null value since one fixation is short enough that it shouldn't have a major impact on analysis
gaze_copy["duration"] = gaze_copy["next time"] - gaze_copy["ts"]
gaze_copy["duration(micro)"] = gaze_copy["duration"].apply(lambda x: x.microseconds)
gaze_copy["duration(s)"] = gaze_copy["duration(micro)"] / 1000000
gaze_copy.drop("duration(micro)", axis=1)
feature_time = gaze_copy.groupby("tag")["duration(s)"].sum()
total_time = gaze_copy["duration(s)"].sum()
percent_time = (feature_time / total_time) * 100

# Records mean duration of looking at each feature
feature_freq = gaze_copy["tag"].value_counts()
features = pd.concat([feature_time, feature_freq], axis=1)
features["mean duration"] = features["duration(s)"] / features["count"]

# Records individual streak durations for each feature
gaze_copy["start of streak"] = gaze_copy["tag"].ne(gaze_copy["tag"].shift())
gaze_copy["streak id"] = gaze_copy["start of streak"].cumsum()
gaze_copy["streak count"] = gaze_copy.groupby("streak id").cumcount() + 1
gaze_copy["end of streak"] = gaze_copy["start of streak"].shift(-1, fill_value=True)
gaze_copy.loc[gaze_copy["start of streak"], ["start time"]] = gaze_copy["ts"]
gaze_copy.loc[gaze_copy["end of streak"], ["end time"]] = gaze_copy["next time"]
streak_time = gaze_copy.groupby("streak id").apply(
    lambda x: x["end time"][-1] - x["start time"][0]
)
streak_time = pd.DataFrame(streak_time)
streak_tag = pd.merge(
    streak_time,
    gaze_copy[["streak id", "tag"]].drop_duplicates(subset="streak id"),
    left_on="streak id",
    right_on="streak id",
    how="left",
)
streak_tag.rename(columns={0: "duration(s)"}, inplace=True)
streak_tag["duration(s)"] = streak_tag["duration(s)"].dt.microseconds / 1000000
streak_tag.set_index("tag")

# Determines max streak by feature
max_streak = streak_tag.groupby("tag").max()

# Records amplitude of saccades by feature
gaze_copy["saccade time(s)"] = gaze_copy["duration(s)"]
gaze_copy["change x"] = gaze_copy["next x"] - gaze_copy["gaze x [px]"]
gaze_copy["change y"] = gaze_copy["next y"] - gaze_copy["gaze y [px]"]
gaze_copy["saccade distance"] = np.sqrt(
    (gaze_copy["change x"] ** 2) + (gaze_copy["change y"] ** 2)
)

# Direction of saccades
gaze_copy["direction x"] = gaze_copy["change x"].apply(
    lambda x: "east" if x > 0 else "west"
)
gaze_copy["direction y"] = gaze_copy["change y"].apply(
    lambda y: "north" if y > 0 else "south"
)
gaze_copy["saccade direction"] = gaze_copy["direction y"] + gaze_copy["direction x"]
gaze_copy.drop(["direction x", "direction y"], axis=1)
