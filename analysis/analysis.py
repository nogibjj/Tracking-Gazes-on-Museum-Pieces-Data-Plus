"""This script describe statistics of when/where people are looking"""


import sys
import os

path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# prepend parent directory to the system path:
sys.path.insert(0, path)

import pandas as pd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import math
import collections
from collections import Counter
import scipy
from config.config import *
import warnings

warnings.filterwarnings("ignore")
plt.style.use("ggplot")


# Set env variables based on config file
try:
    env = sys.argv[1]
    env_var = eval(env + "_config")
except:
    print("Enter valid env variable. Refer to classes in the config.py file")
    sys.exit()


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


output_folder_path = os.path.join(env_var.OUTPUT_PATH, env_var.ART_PIECE)
output_plots_folder_path = os.path.join(env_var.OUTPUT_PATH, env_var.ART_PIECE, "plots")

if not os.path.exists(output_plots_folder_path):
    os.makedirs(output_plots_folder_path)

all_gaze = pd.read_csv(
    os.path.join(output_folder_path, "all_gaze.csv"), compression="gzip"
)
demographic = pd.read_excel(os.path.join(output_folder_path, "demographic.xlsx"))
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


demographic = demographic[
    [
        "School or degree course",
        "age",
        "education",
        "gender",
        "participant_folder",
    ]
]

gaze_copy = pd.merge(
    all_gaze, demographic, left_on="participant_folder", right_on="participant_folder"
)

gaze_copy = gaze_copy.iloc[:, :-1]  # Knocking out duplicate column

# Timestamps
gaze_copy = gaze_copy.groupby("participant_folder").apply(modify)
gaze_copy["seconds_id"] = gaze_copy["increment_marker"].apply(lambda x: x.seconds)
gaze_copy["ts"] = gaze_copy["timestamp [ns]_for_grouping"].apply(
    lambda x: dt.datetime.fromtimestamp(x / 1000000000)
)

# Durations
gaze_copy["duration"] = gaze_copy["next time"] - gaze_copy["ts"]
gaze_copy["duration(micro)"] = gaze_copy["duration"].apply(lambda x: x.microseconds)
gaze_copy["gaze duration(s)"] = gaze_copy["duration(micro)"] / 1000000
gaze_copy = gaze_copy.drop("duration(micro)", axis=1)
gaze_copy = gaze_copy.assign(row_number=range(len(gaze_copy)))
gaze_copy.reset_index(drop=True, inplace=True)


# Pulls out participants who do not have fixation ids
fixation_count = gaze_copy.groupby("participant_folder")["fixation id"].sum().to_frame()
no_fixations = fixation_count[fixation_count["fixation id"] == 0].reset_index()
null_participants = no_fixations.loc[:, "participant_folder"].to_list()
yes_fixations = fixation_count[fixation_count["fixation id"] != 0].reset_index()
usable_participants = yes_fixations.loc[:, "participant_folder"].to_list()

if len(null_participants) > 0:
    print(
        f"There are {len(null_participants)} participants that do not have any fixations."
    )
    if not os.path.exists("quality_control/usable_participants"):
        os.makedirs("quality_control/usable_participants")

    with open("quality_control/usable_participants/error_participants.txt", "w") as f:
        f.write(
            "These are the list of participants in the final csv without fixations:"
        )
        for participant in null_participants:
            f.write(participant + "\n")

        f.write("The number of participants with usable data")


gaze_fixed = gaze_copy[~gaze_copy["participant_folder"].isin(null_participants)]
gaze_fixed["time elapsed(s)"] = gaze_fixed.groupby("participant_folder")[
    "duration"
].cumsum()

# Saccade dataframe
fixation_null = gaze_fixed[gaze_fixed["fixation id"].isnull()]
id = gaze_fixed.groupby(["participant_folder", "fixation id"])
fixation = (
    pd.concat([id.head(1), id.tail(1)])
    .drop_duplicates()
    .sort_values("fixation id")
    .reset_index(drop=True)
)
gaze_saccades = pd.concat([fixation_null, fixation])
gaze_saccades = gaze_saccades.sort_values("row_number").reset_index(drop=True)

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
        "row_number",
        "saccade duration(s)",
        "change x",
        "change y",
        "saccade distance",
        "saccade direction",
    ]
]

gaze_saccades = pd.merge(
    gaze_saccades, saccade_calc, left_on="row_number", right_on="row_number", how="left"
)
id2 = gaze_saccades.groupby(["participant_folder", "fixation id"])
rearrange = gaze_saccades[gaze_saccades["fixation id"].isnull()]
gaze_saccades = pd.concat([id2.tail(1)])
gaze_saccades = pd.concat([rearrange, gaze_saccades])
gaze_saccades = gaze_saccades.sort_values("row_number").reset_index(drop=True)

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

# Saccades per second
saccade_time = gaze_saccades
saccade_time.dropna(subset="fixation id", inplace=True)

m = saccade_time["fixation id"].isna()
s = m.cumsum()
N = 2
saccade_time["new"] = s.map(s[~m].value_counts()).ge(N) & ~m
saccade_time["new1"] = saccade_time["new"].shift(-1)

mask = (saccade_time["new"] == True) & (saccade_time["new1"] == True)
sac_change = saccade_time[mask]
sac_change = sac_change.set_index(sac_change.index + 0.5)
sac_change.loc[:] = np.nan
saccade_time = pd.concat([saccade_time, sac_change], sort=False).drop(
    ["new", "new1"], axis=1
)
saccade_time.sort_index(inplace=True)
saccade_time = saccade_time.reset_index().drop("level_0", axis=1)
saccade_time["participant_folder"] = saccade_time["participant_folder"].fillna(
    method="ffill"
)

sac_per_time = saccade_time[
    ["participant_folder", "fixation id", "time elapsed(s)"]
].reset_index(drop=True)
sac_per_time["fixation id"].fillna("saccade", inplace=True)
sac_per_time["time elapsed(s)"].fillna(method="ffill", inplace=True)
sac_per_time["fixation id"] = sac_per_time["fixation id"].astype(str)

sac_per_sec = (
    sac_per_time.groupby(
        ["participant_folder", pd.Grouper(key="time elapsed(s)", freq="1S")]
    )["fixation id"]
    .apply(lambda x: x[x.str.contains("saccade")].count())
    .to_frame()
)
sac_per_sec = sac_per_sec.rename({"fixation id": "saccade freq"}, axis=1).reset_index()
sac_per_sec_mean = (
    sac_per_sec.groupby("participant_folder")["saccade freq"].mean().to_frame()
)
sac_per_sec_mean.reset_index(inplace=True)

# Most common saccade direction
direction_mode = (
    gaze_saccades.groupby("participant_folder")["saccade direction"]
    .agg(pd.Series.mode)
    .to_frame()
)
direction_mode.reset_index(inplace=True)

# Mean saccade distance
sac_distance_mean = (
    gaze_saccades.groupby("participant_folder")["saccade distance"].mean().to_frame()
)
sac_distance_mean.reset_index(inplace=True)

# Mean saccade duration
sac_dur_mean = (
    gaze_saccades.groupby("participant_folder")["saccade duration(s)"].mean().to_frame()
)
sac_dur_mean.reset_index(inplace=True)


# Fixation dataframe
gaze_fixation = gaze_fixed[~gaze_fixed["fixation id"].isnull()].copy()
gaze_fixation["fixation id"].value_counts(dropna=False)
gaze_fixation = gaze_fixation.drop("row_number", axis=1).reset_index(drop=True)

# Fixation calculations
fix_durations = gaze_fixation.groupby(["participant_folder", "fixation id"])[
    "gaze duration(s)"
].sum()
fix_durations = fix_durations.to_frame()
fix_durations = fix_durations.rename(
    {"gaze duration(s)": "fixation duration(s)"}, axis=1
)
fix_durations.reset_index(inplace=True)
gaze_fixation = pd.merge(
    gaze_fixation,
    fix_durations,
    left_on=["participant_folder", "fixation id"],
    right_on=["participant_folder", "fixation id"],
    how="left",
)

fix_mean = fix_durations.groupby(["participant_folder"])["fixation duration(s)"].mean()
fix_mean = fix_mean.to_frame()
fix_mean = fix_mean.rename({"fixation duration(s)": "mean fix duration(s)"}, axis=1)

overall_mean = fix_mean["mean fix duration(s)"].mean()

# Fixations per second for each participant
fix_per_time = gaze_fixed[
    ["participant_folder", "fixation id", "time elapsed(s)"]
].reset_index(drop=True)

fix_per_sec = (
    fix_per_time.groupby(
        ["participant_folder", pd.Grouper(key="time elapsed(s)", freq="1S")]
    )["fixation id"]
    .nunique()
    .to_frame()
)
fix_per_sec = fix_per_sec.rename({"fixation id": "fixation freq"}, axis=1).reset_index()
fix_per_sec_mean = (
    fix_per_sec.groupby("participant_folder")["fixation freq"].mean().to_frame()
)
fix_per_sec_mean.reset_index(inplace=True)

# Most fixated on feature
most_fix_gen = (
    gaze_fixation.groupby("participant_folder")["general tag"]
    .agg(pd.Series.mode)
    .to_frame()
)
most_fix_gen.reset_index(inplace=True)

most_fix_tag = (
    gaze_fixation.groupby("participant_folder")["tag"].agg(pd.Series.mode).to_frame()
)
most_fix_tag.reset_index(inplace=True)

# Fixation time per feature
tag_fix = (
    gaze_fixation.groupby(["participant_folder", "tag"])["gaze duration(s)"]
    .sum()
    .to_frame()
)
tag_fix.reset_index(inplace=True)
tag_fix_avg = tag_fix.groupby("tag")["gaze duration(s)"].mean()

gen_tag_fix = (
    gaze_fixation.groupby(["participant_folder", "general tag"])["gaze duration(s)"]
    .sum()
    .to_frame()
)
gen_tag_fix.reset_index(inplace=True)
gen_tag_fix_avg = gen_tag_fix.groupby("general tag")["gaze duration(s)"].mean()

# First and last things participants fixated on and for how long
g = gaze_fixation.groupby("participant_folder")

first = pd.concat([g.head(1)]).sort_values("participant_folder")
first = first[["participant_folder", "general tag", "tag", "fixation duration(s)"]]
first.reset_index(inplace=True)

last = pd.concat([g.tail(1)]).sort_values("participant_folder")
last = last[["participant_folder", "general tag", "tag", "fixation duration(s)"]]
last.reset_index(inplace=True)


# Most common sequence
def most_common(words, n):
    last_group = len(words) - (n - 1)

    groups = (tuple(words[start : start + n]) for start in range(last_group))

    return Counter(groups).most_common()


uniqueid = gaze_fixation.groupby(["participant_folder", "fixation id"])
unique_fix = pd.concat([uniqueid.first()]).reset_index()
mode_fix = uniqueid["tag"].agg(pd.Series.mode).to_frame().reset_index()
unique_fix["tag"] = mode_fix["tag"].astype(str)
unique_fix = unique_fix[["participant_folder", "tag"]]

participants = dict()
for k, v in unique_fix.groupby("participant_folder")["tag"]:
    participants[k] = v

for k, v in participants.items():
    participants[k] = most_common(v, 3)

for k, v in participants.items():
    participants[k] = v[0]

common_seq = pd.DataFrame.from_dict(participants).transpose().reset_index()
common_seq.columns = ["participant_folder", "sequence", "count"]

# Putting together cumulative analysis dataframe
analysis = fix_mean.reset_index()
analysis["fixation freq"] = fix_per_sec_mean["fixation freq"]
analysis["most fixated gen"] = most_fix_gen["general tag"]
analysis["most fixated tag"] = most_fix_tag["tag"]
analysis["first fixation"] = first["tag"]
analysis["first fix time"] = first["fixation duration(s)"]
analysis["last fixation"] = last["tag"]
analysis["last fix time"] = last["fixation duration(s)"]
analysis["mean sac duration(s)"] = sac_dur_mean["saccade duration(s)"]
analysis["mean sac distance"] = sac_distance_mean["saccade distance"]
analysis["mode sac direction"] = direction_mode["saccade direction"]
analysis["saccade freq"] = sac_per_sec_mean["saccade freq"]
analysis["most common sequence"] = common_seq["sequence"]
analysis["sequence count"] = common_seq["count"]

# First time participants fixated on each feature
genid = gaze_fixation.groupby(["participant_folder", "general tag"])
first_gen = pd.concat([genid.first()])
first_gen.reset_index(inplace=True)
first_gen = first_gen[["participant_folder", "general tag", "time elapsed(s)"]]

tagid = gaze_fixation.groupby(["participant_folder", "tag"])
first_tag = pd.concat([tagid.first()])
first_tag.reset_index(inplace=True)
first_tag = first_tag[["participant_folder", "tag", "time elapsed(s)"]]

# Visualizations (boxplot)
# Time spent on each feature
tag_time = (
    gaze_fixation.groupby(["participant_folder", "tag"])["gaze duration(s)"]
    .sum()
    .to_frame()
    .reset_index()
)
tag_time.boxplot(column="gaze duration(s)", by="tag", vert=False)
plt.xlabel("Duration (s)")
plt.ylabel("Feature")
plt.title("Fixation Duration by Feature (All Participants)")
plt.suptitle("")
plt.show()

# Demographics
# Add demographic data to analysis dataframe
analysis = pd.merge(
    analysis,
    demographic,
    left_on="participant_folder",
    right_on="participant_folder",
    how="left",
)
analysis.sort_values("age", inplace=True)
analysis["age group"] = pd.cut(
    analysis["age"], bins=6, right=True, precision=0, include_lowest=True
)

# Age
fix_dur_age = analysis.groupby("age group")["mean fix duration(s)"].mean()
fix_per_sec_age = analysis.groupby("age group")["fixation freq"].mean()
first_fix_dur_age = analysis.groupby("age group")["first fix time"].mean()
last_fix_dur_age = analysis.groupby("age group")["last fix time"].mean()
sac_dur_age = analysis.groupby("age group")["mean sac duration(s)"].mean()
sac_dist_age = analysis.groupby("age group")["mean sac distance"].mean()
sac_per_sec_age = analysis.groupby("age group")["saccade freq"].mean()

# Education
fix_dur_edu = analysis.groupby("education")["mean fix duration(s)"].mean()
fix_per_sec_edu = analysis.groupby("education")["fixation freq"].mean()
first_fix_dur_edu = analysis.groupby("education")["first fix time"].mean()
last_fix_dur_edu = analysis.groupby("education")["last fix time"].mean()
sac_dur_edu = analysis.groupby("education")["mean sac duration(s)"].mean()
sac_dist_edu = analysis.groupby("education")["mean sac distance"].mean()
sac_per_sec_edu = analysis.groupby("education")["saccade freq"].mean()

# Gender
fix_dur_gender = analysis.groupby("gender")["mean fix duration(s)"].mean()
fix_per_sec_gender = analysis.groupby("gender")["fixation freq"].mean()
first_fix_dur_gender = analysis.groupby("gender")["first fix time"].mean()
last_fix_dur_gender = analysis.groupby("gender")["last fix time"].mean()
sac_dur_gender = analysis.groupby("gender")["mean sac duration(s)"].mean()
sac_dist_gender = analysis.groupby("gender")["mean sac distance"].mean()
sac_per_sec_gender = analysis.groupby("gender")["saccade freq"].mean()

# Demographic visualizations (boxplots)
vars_list = [
    ["mean fix duration(s)", "age group"],
    ["mean fix duration(s)", "education"],
    ["mean fix duration(s)", "gender"],
    ["first fix time", "age group"],
    ["first fix time", "education"],
    ["first fix time", "gender"],
    ["last fix time", "age group"],
    ["last fix time", "education"],
    ["last fix time", "gender"],
    ["fixation freq", "age group"],
    ["fixation freq", "education"],
    ["fixation freq", "gender"],
    ["mean sac duration(s)", "age group"],
    ["mean sac duration(s)", "education"],
    ["mean sac duration(s)", "gender"],
    ["mean sac distance", "age group"],
    ["mean sac distance", "education"],
    ["mean sac distance", "gender"],
    ["saccade freq", "age group"],
    ["saccade freq", "education"],
    ["saccade freq", "gender"],
]


for vars in vars_list:
    analysis.boxplot(column=vars[0], by=vars[1])
    plt.xlabel(vars[0])
    plt.ylabel(vars[1])
    plt.title(f"Plotting {vars[0]} by {vars[1]}")
    plt.suptitle("")
    path = os.path.join(output_plots_folder_path, f"{vars[0]}_{vars[1]}.png")
    plt.savefig(path)

# Checking for fixations in participants who don't have registered fixation IDs
gaze_null = gaze_copy[gaze_copy["participant_folder"].isin(null_participants)]
gaze_null.drop("row_number", axis=1, inplace=True)

qualify = gaze_null[gaze_null["gaze duration(s)"] > 0.06]
qualify["change_x"] = qualify["next x"] - qualify["gaze x [px]"]
qualify["change_y"] = qualify["next y"] - qualify["gaze y [px]"]
qualify["distance"] = np.sqrt((qualify["change_x"] ** 2) + (qualify["change_y"] ** 2))
qualify["velocity"] = qualify["distance"] / qualify["gaze duration(s)"]
qualify["angle(r)"] = qualify.apply(
    lambda x: math.atan2(x.change_y, x.change_x), axis=1
)
