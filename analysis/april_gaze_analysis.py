"""This script describe statistics of when/where people are looking"""

import pandas as pd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt


# Groupby function
def modify(df):
    df.reset_index(inplace=True, drop=True)
    df["ts"] = pd.to_datetime(df["ts"])
    baseline = df["ts"][0]
    df["increment_marker"] = df["ts"] - baseline
    df["next time"] = df["ts"].shift(-1)
    df["next x"] = df["gaze x [px]"].shift(-1)
    df["next y"] = df["gaze y [px]"].shift(-1)
    df = df[:-1].copy()
    return df


import sys

sys.path.insert(0, "..")
print(sys.path)
all_gaze = pd.read_csv("../data/all_gaze.csv", compression="gzip")

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

all_gaze["general tag"] = all_gaze["tag"].apply(lambda x: x.split(" ")[-1])

demographic = pd.read_excel("../data/demographic.xlsx")
demographic = demographic[
    [
        "School or degree course",
        "Age",
        "Educational Qualification",
        "sesso",
        "codice_eyetr_museo",
    ]
]

gaze_copy = pd.merge(
    all_gaze, demographic, left_on="participant_folder", right_on="codice_eyetr_museo"
)

gaze_copy = gaze_copy.iloc[:, :-1]  # Knocking out duplicate column


# Pulls out participants who do not have fixation ids
# Excludes them from analysis
fixation_count = gaze_copy.groupby("participant_folder").sum()
no_fixations = fixation_count[fixation_count["fixation id"] == 0].reset_index()
null_participants = no_fixations.loc[:, "participant_folder"].to_list()
gaze_fixed = gaze_copy[~gaze_copy["participant_folder"].isin(null_participants)]


# Time spent on each feature
gaze_fixed = gaze_fixed.groupby("participant_folder").apply(modify)
gaze_fixed["seconds_id"] = gaze_fixed["increment_marker"].apply(lambda x: x.seconds)
gaze_fixed["ts"] = gaze_fixed["timestamp [ns]_for_grouping"].apply(
    lambda x: dt.datetime.fromtimestamp(x / 1000000000)
)

# Durations
gaze_fixed["duration"] = gaze_fixed["next time"] - gaze_fixed["ts"]
gaze_fixed["duration(micro)"] = gaze_fixed["duration"].apply(lambda x: x.microseconds)
gaze_fixed["gaze duration(s)"] = gaze_fixed["duration(micro)"] / 1000000
gaze_fixed = gaze_fixed.drop("duration(micro)", axis=1)

# Fixation dataframe
gaze_fixed = gaze_fixed.assign(row_number=range(len(gaze_fixed)))
gaze_fixation = gaze_fixed[~gaze_fixed["fixation id"].isnull()].copy()
gaze_fixation["fixation id"].value_counts(dropna=False)

# Saccade dataframe
fixation_null = gaze_fixed[gaze_fixed["fixation id"].isnull()]
id = gaze_fixed.groupby("fixation id")
fixation = (
    pd.concat([id.head(1), id.tail(1)])
    .drop_duplicates()
    .sort_values("fixation id")
    .reset_index(drop=True)
)
gaze_saccades = pd.concat([fixation_null, fixation])
gaze_saccades = gaze_saccades.sort_values("row_number").reset_index(drop=True)
gaze_saccades = gaze_saccades.drop("row_number", axis=1)

# Saccade calculations
saccade_calc = gaze_saccades[gaze_saccades["fixation id"].notnull()]
saccade_calc = saccade_calc.groupby("participant_folder").apply(modify)

saccade_calc["s duration"] = saccade_calc["next time"] - saccade_calc["ts"]
saccade_calc["s duration(micro)"] = saccade_calc["s duration"].apply(
    lambda x: x.microseconds
)
saccade_calc["saccade duration(s)"] = saccade_calc["s duration(micro)"] / 1000000
saccade_calc = saccade_calc.drop("s duration(micro)", axis=1)

saccade_calc["change x"] = saccade_calc["next x"] - saccade_calc["gaze x [px]"]
saccade_calc["change y"] = saccade_calc["next y"] - saccade_calc["gaze y [px]"]
saccade_calc["saccade distance"] = np.sqrt(
    (saccade_calc["change x"] ** 2) + (saccade_calc["change y"] ** 2)
)

conditions = [
    (saccade_calc["change x"] > 0) & (saccade_calc["change y"] == 0),
    (saccade_calc["change x"] < 0) & (saccade_calc["change y"] == 0),
    (saccade_calc["change x"] == 0) & (saccade_calc["change y"] > 0),
    (saccade_calc["change x"] == 0) & saccade_calc["change y"] < 0,
    (saccade_calc["change x"] > 0) & (saccade_calc["change y"] > 0),
    (saccade_calc["change x"] > 0) & (saccade_calc["change y"] < 0),
    (saccade_calc["change x"] < 0) & (saccade_calc["change y"] > 0),
    (saccade_calc["change x"] < 0) & (saccade_calc["change y"] < 0),
]

choices = [
    "east",
    "west",
    "north",
    "south",
    "northeast",
    "southeast",
    "northwest",
    "southwest",
]

saccade_calc["saccade direction"] = np.select(conditions, choices, "none")
saccade_calc = saccade_calc.reset_index(drop=True)
saccade_calc = saccade_calc[
    [
        "index",
        "saccade duration(s)",
        "change x",
        "change y",
        "saccade distance",
        "saccade direction",
    ]
]

gaze_saccades = pd.merge(
    gaze_saccades, saccade_calc, left_on="index", right_on="index", how="left"
)

# Insert empty row if saccade isn't recorded
m = gaze_saccades["fixation id"].isna()
s = m.cumsum()
N = 2
gaze_saccades["new"] = s.map(s[~m].value_counts()).ge(N) & ~m
gaze_saccades["new1"] = gaze_saccades["new"].shift(-1)

mask = (gaze_saccades["new"] == True) & (gaze_saccades["new1"] == True)
saccade_change = gaze_saccades[mask]
saccade_change = saccade_change.set_index(saccade_change.index + 0.5)
saccade_change.loc[:] = np.nan
gaze_saccades = pd.concat([gaze_saccades, saccade_change], sort=False).drop(
    ["new", "new1"], axis=1
)
gaze_saccades.sort_index(inplace=True)
gaze_saccades = gaze_saccades.reset_index().drop("level_0", axis=1)
gaze_saccades["participant_folder"] = gaze_saccades["participant_folder"].fillna(
    method="ffill"
)

# Organizing saccade calcuations
gaze_saccades["saccade direction"] = gaze_saccades["saccade direction"].fillna(
    method="ffill"
)
gaze_saccades.loc[gaze_saccades["fixation id"].notnull(), "saccade direction"] = np.nan
gaze_saccades["saccade duration(s)"] = gaze_saccades["saccade duration(s)"].fillna(
    method="ffill"
)
gaze_saccades.loc[
    gaze_saccades["fixation id"].notnull(), "saccade duration(s)"
] = np.nan


# Dataframe for testing, delete later
saccade_test = gaze_saccades

"""
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
mode_ft_age.to_csv("mode_feature_by_age.csv", index=False)"""


# Commented out for now, leaving the general section at the end
# # The same applies here, but for a general feature.
# feature_time_general = gaze_copy.groupby("general tag")["duration(s)"].sum()
# total_time = gaze_copy["duration(s)"].sum()
# percent_time_general = (feature_time_general / total_time) * 100
