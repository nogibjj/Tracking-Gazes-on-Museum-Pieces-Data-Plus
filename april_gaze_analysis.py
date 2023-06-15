"""This script describe statistics of when/where people are looking"""

import pandas as pd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
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


# Most frequently looked at/returned to feature
all_gaze = pd.read_csv("all_gaze.csv", compression="gzip")

all_gaze = fake_tagger(all_gaze)
all_gaze.reset_index(drop=True, inplace=True)
mode = all_gaze["tag"].mode()


def participant_folder_corrector(input_string):
    """Add a 0 after the underscore in participant_folder"""
    import re

    try:
        target_for_replacement = re.findall(pattern="\d+(_\d+).*", string=input_string)[
            0
        ]

    except:
        return input_string

    if len(target_for_replacement) >= 3:
        return input_string

    else:
        return input_string.replace("_", "_0")


demographic = pd.read_excel("demographic.xlsx")
demographic["codice_eyetr_museo"] = demographic["codice_eyetr_museo"].fillna("none")
demographic["codice_eyetr_museo"] = demographic["codice_eyetr_museo"].apply(
    lambda x: participant_folder_corrector(x)
)
all_gaze["codice_eyetr_museo"] = all_gaze["participant_folder"].apply(
    lambda x: participant_folder_corrector(x)
)

gaze_copy = pd.merge(
    all_gaze,
    demographic[
        [
            "School or degree course",
            "Age",
            "Educational Qualification",
            "sesso",
            "codice_eyetr_museo",
        ]
    ],
    left_on="participant_folder",
    right_on="codice_eyetr_museo",
    how="left",
)

gaze_copy = gaze_copy.iloc[:, :-1]

# Percent time spent on each feature
gaze_copy["ts"] = gaze_copy["timestamp [ns]"].apply(
    lambda x: dt.datetime.fromtimestamp(x / 1000000000)
)
gaze_copy = gaze_copy.groupby(["participant_folder"]).apply(modify)
gaze_copy["seconds_id"] = gaze_copy["increment_marker"].apply(lambda x: x.seconds)

# Drop last row to get rid of null value since one fixation is short enough that it shouldn't have a major impact on analysis
gaze_copy["duration"] = gaze_copy["next time"] - gaze_copy["ts"]
gaze_copy["duration(micro)"] = gaze_copy["duration"].apply(lambda x: x.microseconds)
gaze_copy["duration(s)"] = gaze_copy["duration(micro)"] / 1000000
gaze_copy.drop("duration(micro)", axis=1)
feature_time = gaze_copy.groupby("tag")["duration(s)"].sum()
total_time = gaze_copy["duration(s)"].sum()
percent_time = (feature_time / total_time) * 100

# Percent time spent on each feature in women
gaze_f = gaze_copy[gaze_copy["sesso"] == "f"]
feature_f = gaze_f.groupby("tag")["duration(s)"].sum()
total_f = gaze_f["duration(s)"].sum()
percent_f = (feature_f / total_f) * 100

# Percent time spent on each feature in men
gaze_m = gaze_copy[gaze_copy["sesso"] == "m"]
feature_m = gaze_m.groupby("tag")["duration(s)"].sum()
total_m = gaze_m["duration(s)"].sum()
percent_m = (feature_m / total_m) * 100

# Mean fixation duration of looking at each feature
feature_freq = gaze_copy["tag"].value_counts()
features = pd.concat([feature_time, feature_freq], axis=1)
features["mean fix duration(s)"] = features["duration(s)"] / features["count"]

# Mean duration of looking at each feature
features["mean duration(s)"] = (
    features["duration(s)"] / gaze_copy["participant_folder"].nunique()
)

# Individual streak durations for each feature
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

# Max streak by feature
max_streak = streak_tag.groupby("tag").max()

# Mean streak by feature
mean_streak = streak_tag.groupby("tag").mean()

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

# Total % time spent on each feature plotted (all)
percent_time = percent_time.to_frame()
percent_time = percent_time.reset_index()
percent_time.columns = ["tag", "% time"]
plt.bar(percent_time["tag"], percent_time["% time"])
plt.xlabel("Feature")
plt.ylabel("% Time Spent Looking")
plt.title("% Time Spent Looking at Each Feature (All Participants)")
plt.savefig("%_time_plot.png")
plt.show()

# Mean streak duration for each feature plotted (all)
mean_streak = mean_streak.reset_index()
plt.bar(mean_streak["tag"], mean_streak["duration(s)"])
plt.xlabel("Feature")
plt.ylabel("Duration(s)")
plt.title("Mean Streak Duration by Feature (All Participants)")
plt.savefig("mean_streak_plot.png")
plt.show()

# Mean duration spent looking at each feature plotted (all)
features = features.reset_index()
plt.bar(features["tag"], features["mean duration(s)"])
plt.xlabel("Feature")
plt.ylabel("Duration(s)")
plt.title("Mean Duration Spent Looking at Each Feature (All Participants)")
plt.savefig("mean_dur_plot.png")
plt.show()

# Percent time spent on each feature in men vs. women plotted
percent_f = percent_f.to_frame()
percent_f = percent_f.reset_index()
percent_f.columns = ["tag", "Women"]
percent_m = percent_m.to_frame()
percent_m = percent_m.reset_index()
percent_m.columns = ["tag", "Men"]
percent_fvm = pd.merge(percent_m, percent_f, on="tag")
percent_fvm.plot(
    x="tag",
    y=["Men", "Women"],
    kind="bar",
    title="% Time Spent Looking at Each Feature in Men vs. Women",
    xlabel="Feature",
    ylabel="% Time Spent",
)
plt.savefig("%_time_mvf.png")

# Most looked at feature by age group table
gaze_copy["age group"] = pd.cut(
    gaze_copy["Age"], bins=6, right=True, precision=0, include_lowest=True
)
ages = ["17-24", "25-32", "33-40", "41-48", "49-56", "57-64"]
mode_by_age = gaze_copy.groupby(["age group"])["tag"].agg(pd.Series.mode)
mode_ft_age = pd.DataFrame(ages, columns=["Age"])
mode_ft_age["Feature"] = mode_by_age.tolist()
mode_ft_age.to_csv("mode_feature_by_age.csv", index=False)
