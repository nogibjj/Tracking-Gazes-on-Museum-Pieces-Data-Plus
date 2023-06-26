"""This script describe statistics of when/where people are looking"""

import pandas as pd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt


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


# Feature with highest number of fixations
all_gaze = pd.read_csv("all_gaze.csv", compression="gzip")

all_gaze.reset_index(drop=True, inplace=True)


# tag mapping to words, to be erased later on
mapping = {
    "r base": "right base",
    "noise": "noise",
    "m r hand": "male right hand",
    "m l hand": "male left hand",
    "l base": "left base",
    "m feet": "male feet",
    "m face": "male face",
    "m chest": "male chest",
    "lights": "lights",
    "guy": "corner stranger",
    "f r hand": "female right hand",
    "f l hand": "female left hand",
    "f feet": "female feet",
    "f face": "female face",
    "f chest": "female chest",
    "crack": "asymmetric fissure",
    "bg": "background",
}

all_gaze["tag"] = all_gaze["tag"].map(mapping)

modes = all_gaze["tag"].value_counts()

all_gaze["general tag"] = all_gaze["tag"].apply(lambda x: x.split(" ")[-1])


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


def participant_corrector(input_string):
    """Fixing the 1 digit issue after the underscore"""
    items = input_string.split("_")  # attach  maxsplit later
    section_of_interest = items[-1]
    count = 0
    for char in section_of_interest:
        if char.isdigit():
            count += 1

        else:
            pass

    if count >= 2:
        return input_string

    else:
        return input_string.replace("_", "_0")


demographic = pd.read_excel("demographic.xlsx")
demographic["codice_eyetr_museo"] = demographic["codice_eyetr_museo"].fillna("none")
demographic["codice_eyetr_museo"] = demographic["codice_eyetr_museo"].apply(
    lambda x: participant_corrector(x)
)
all_gaze["codice_eyetr_museo"] = all_gaze["participant_folder"].apply(
    lambda x: participant_corrector(x)
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

gaze_copy = gaze_copy.iloc[:, :-1]  # Knocking out duplicate column

# fixation analysis
gaze_copy = gaze_copy[~gaze_copy["fixation id"].isnull()].copy()
gaze_copy["fixation id"].value_counts(dropna=False)

# Percent time spent on each feature
gaze_copy["ts"] = gaze_copy["timestamp [ns]_for_grouping"].apply(
    lambda x: dt.datetime.fromtimestamp(x / 1000000000)
)
gaze_copy = gaze_copy.groupby(["participant_folder"]).apply(modify)
gaze_copy["seconds_id"] = gaze_copy["increment_marker"].apply(lambda x: x.seconds)


# This block answers, after summing the durations across all videos
# across all fixations, what percent of time was spent fixating
# on a specific feature
# Drop last row to get rid of null value since one fixation is short enough that it shouldn't have a major impact on analysis
gaze_copy["duration"] = gaze_copy["next time"] - gaze_copy["ts"]
gaze_copy["duration(micro)"] = gaze_copy["duration"].apply(lambda x: x.microseconds)
gaze_copy["duration(s)"] = gaze_copy["duration(micro)"] / 1000000
gaze_copy.drop("duration(micro)", axis=1)
feature_time = gaze_copy.groupby("tag")["duration(s)"].sum()
fixation_time = gaze_copy.groupby("tag")["fixation id"].sum()
total_time = gaze_copy["duration(s)"].sum()
percent_time = (feature_time / total_time) * 100


# Same extension for women
# Percent time spent on each feature in women
gaze_f = gaze_copy[gaze_copy["sesso"] == "f"]
feature_f = gaze_f.groupby("tag")["duration(s)"].sum()
total_f = gaze_f["duration(s)"].sum()
percent_f = (feature_f / total_f) * 100

# Same extension for men
# Percent time spent on each feature in men
gaze_m = gaze_copy[gaze_copy["sesso"] == "m"]
feature_m = gaze_m.groupby("tag")["duration(s)"].sum()
total_m = gaze_m["duration(s)"].sum()
percent_m = (feature_m / total_m) * 100


# Saving percent time metrics to csv
round(percent_time, 2).to_csv("percent_time.csv")
round(percent_f, 2).to_csv("percent_time_f.csv")
round(percent_m, 2).to_csv("percent_time_m.csv")


##### Mean fixation duration of looking at each feature

# feature_freq = gaze_copy["tag"].value_counts() # FIX ONLY THIS

# feature time adds the durations
# feature freq is not the right count to produce this average
# Here is the current fix, which counts unique fixations on
# the target feature
feature_freq = gaze_copy.groupby(["tag"])["fixation id"].nunique()
features = pd.concat([feature_time, feature_freq], axis=1)
features["mean fix duration(s)"] = features["duration(s)"] / features["fixation id"]

# Saving mean fixation duration metrics to csv
round(features, 2).to_csv("mean_fix_duration.csv")
new_features = round(features, 2)
# new_features = new_features.iloc[:,-1]
features.to_csv("mean_fix_duration.csv")

# make an horizontal bar plot of the mean fixation duration
# with the numerical values near the bars
new_features.plot(
    kind="barh",
    title="Mean Fixation Duration by Feature",
    xlabel="Feature",
    ylabel="Duration(s)",
)

import matplotlib.pyplot as plt

x = [x for x in new_features.index]
y = [y for y in new_features["mean fix duration(s)"]]
plt.barh(x, y)
plt.title("Mean Fixation Duration (s) by Feature")
for index, value in enumerate(y):
    plt.text(value, index, str(value))

plt.show()


import matplotlib.pyplot as plt

x = [x for x in fixation_time.index]
y = [int(y) for y in fixation_time.values]
plt.style.use("ggplot")
plt.barh(x, y)
plt.title("Fixation Count by Feature")
for index, value in enumerate(y):
    plt.text(value, index, str(value))

plt.show()

# The following block would be
# if fixed the mean durations of the fixations per the features per participant

# # Mean duration of looking at each feature
# features["mean duration(s)"] = (
#     features["duration(s)"] / gaze_copy["participant_folder"].nunique()
# )

# Evaluating if this section can be modified to answer a question
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

# Number of streaks per feature

# Amplitude of saccades by feature
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
gaze_plot = (
    gaze_copy[["tag", "duration(s)", "sesso"]].reset_index().drop("level_1", axis=1)
)
time_per_participant = gaze_plot.groupby(["participant_folder", "tag"]).agg(
    {"duration(s)": "sum"}
)
time_per_participant["percentage"] = (
    time_per_participant["duration(s)"] / total_time
) * 100
total_percent_plot = time_per_participant.boxplot(
    column="duration(s)", by="tag", grid=False, figsize=(10, 10), vert=False
)
plt.title("% Time Spent Looking at Each Feature (All Participants)")
plt.suptitle("")
plt.xlabel("Feature")
plt.ylabel("% Time Spent Looking")
plt.savefig("%_time_plot.png")
plt.show()

# Mean streak duration for each feature plotted (all)
mean_streak_plot = streak_tag.boxplot(
    column="duration(s)", by="tag", grid=False, figsize=(10, 10), vert=False
)
plt.title("Mean Streak Duration by Feature (All Participants)")
plt.suptitle("")
plt.xlabel("Feature")
plt.ylabel("Duration(s)")
plt.savefig("mean_streak_plot.png")
plt.show()

# Mean duration spent looking at each feature plotted (all)
mean_dur_plot = time_per_participant.boxplot(column="duration(s)", by="tag", grid=False)
plt.title("Mean Duration Spent Looking at Each Feature (All Participants)")
plt.suptitle("")
plt.xlabel("Feature")
plt.ylabel("Duration(s)")
plt.savefig("mean_dur_plot.png")
plt.show()

# Percent time spent on each feature in men vs. women plotted
time_women = gaze_plot[gaze_plot["sesso"] == "f"]
time_per_woman = time_women.groupby(["participant_folder", "tag"]).agg(
    {"duration(s)": "sum"}
)
time_per_woman["percentage"] = (time_per_woman["duration(s)"] / total_time) * 100
women_percent_plot = time_per_woman.boxplot(column="percentage", by="tag", grid=False)
plt.title("% Time Spent Looking at Each Feature (Women)")
plt.suptitle("")
plt.xlabel("Feature")
plt.ylabel("Duration(s)")
plt.savefig("%_time_women.png")
plt.show()

time_men = gaze_plot[gaze_plot["sesso"] == "m"]
time_per_man = time_men.groupby(["participant_folder", "tag"]).agg(
    {"duration(s)": "sum"}
)
time_per_man["percentage"] = (time_per_man["duration(s)"] / total_time) * 100
men_percent_plot = time_per_man.boxplot(column="percentage", by="tag", grid=False)
plt.title("% Time Spent Looking at Each Feature (Men)")
plt.suptitle("")
plt.xlabel("Feature")
plt.ylabel("Duration(s)")
plt.savefig("%_time_men.png")
plt.show()

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
plt.savefig("%_time_mvf_bar.png")

# Most looked at feature by age group table
gaze_copy["age group"] = pd.cut(
    gaze_copy["Age"], bins=6, right=True, precision=0, include_lowest=True
)
ages = ["17-24", "25-32", "33-40", "41-48", "49-56", "57-64"]
mode_by_age = gaze_copy.groupby(["age group"])["tag"].agg(pd.Series.mode)
mode_ft_age = pd.DataFrame(ages, columns=["Age"])
mode_ft_age["Feature"] = mode_by_age.tolist()
mode_ft_age.to_csv("mode_feature_by_age.csv", index=False)


# Commented out for now, leaving the general section at the end
# # The same applies here, but for a general feature.
# feature_time_general = gaze_copy.groupby("general tag")["duration(s)"].sum()
# total_time = gaze_copy["duration(s)"].sum()
# percent_time_general = (feature_time_general / total_time) * 100
