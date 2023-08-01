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
import seaborn as sns
import math
import collections
from collections import Counter
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

gaze_copy = pd.read_csv(
    os.path.join(output_folder_path, "all_gaze.csv"), compression="gzip"
)

data_folder_path = os.path.join(
    env_var.ROOT_PATH, env_var.ART_PIECE
)  # all the participant folders are here

# Timestamps
gaze_copy["ts"] = gaze_copy["timestamp [ns]"].apply(
    lambda x: dt.datetime.fromtimestamp(x / 1000000000)
)
gaze_copy = gaze_copy.groupby("participant_folder").apply(modify)
gaze_copy["seconds_id"] = gaze_copy["increment_marker"].apply(lambda x: x.seconds)

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
        f"There are {len(null_participants)} participants that do not have registered fixation IDs."
    )

    with open(os.path.join(output_folder_path, "error_participants.txt"), "w") as f:
        f.write(
            "These are the list of participants in the final csv without fixation IDs: \n"
        )
        for participant in null_participants:
            f.write(participant + "\n")

        f.write(
            "The number of participants with usable fixation data, as determined by pupil"
        )

        for participant in usable_participants:
            f.write(participant + "\n")

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
qualify = qualify[["participant_folder", "velocity", "angle(r)"]]
qualify["angle(r)"] = qualify["angle(r)"].apply(abs)
qualify["fixation"] = np.where(
    (qualify["velocity"] < 900.0) & (qualify["angle(r)"] < 1.5), True, False
)

not_null = qualify[qualify["fixation"] == True]
not_null_participants = not_null["participant_folder"]
if len(not_null_participants) > 0:
    not_null.to_excel(
        os.path.join(
            output_folder_path, "qualified participants without fixation IDs.xlsx"
        )
    )
    print(
        f"There are {len(not_null_participants)} participants who have gaze points that qualify for fixations, but do not have registered fixation IDs."
    )

# Create cleaned dataframe
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
sac_per_sec = sac_per_sec.rename(
    {"fixation id": "saccade frequency (hz)"}, axis=1
).reset_index()
sac_per_sec_mean = (
    sac_per_sec.groupby("participant_folder")["saccade frequency (hz)"]
    .mean()
    .to_frame()
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
    {"gaze duration(s)": "fixation duration (s)"}, axis=1
)
fix_durations.reset_index(inplace=True)
gaze_fixation = pd.merge(
    gaze_fixation,
    fix_durations,
    left_on=["participant_folder", "fixation id"],
    right_on=["participant_folder", "fixation id"],
    how="left",
)

fix_mean = fix_durations.groupby(["participant_folder"])["fixation duration (s)"].mean()
fix_mean = fix_mean.to_frame()
fix_mean = fix_mean.rename({"fixation duration (s)": "fixation duration (s)"}, axis=1)

overall_mean = fix_mean["fixation duration (s)"].mean()

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
fix_per_sec = fix_per_sec.rename(
    {"fixation id": "fixation frequency (hz)"}, axis=1
).reset_index()
fix_per_sec_mean = (
    fix_per_sec.groupby("participant_folder")["fixation frequency (hz)"]
    .mean()
    .to_frame()
)
fix_per_sec_mean.reset_index(inplace=True)

# Most fixated on feature
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

# First and last things participants fixated on and for how long
g = gaze_fixation.groupby("participant_folder")

first = pd.concat([g.head(1)]).sort_values("participant_folder")
first = first[["participant_folder", "tag", "fixation duration (s)"]]
first.reset_index(inplace=True)

last = pd.concat([g.tail(1)]).sort_values("participant_folder")
last = last[["participant_folder", "tag", "fixation duration (s)"]]
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
analysis["fixation frequency (hz)"] = fix_per_sec_mean["fixation frequency (hz)"]
analysis["most fixated feature"] = most_fix_tag["tag"]
analysis["first fixation"] = first["tag"]
analysis["first fixation duration (s)"] = first["fixation duration (s)"]
analysis["last fixation"] = last["tag"]
analysis["last fixation duration (s)"] = last["fixation duration (s)"]
analysis["saccade duration(s)"] = sac_dur_mean["saccade duration(s)"]
analysis["saccade distance (px)"] = sac_distance_mean["saccade distance"]
analysis["mode sac direction"] = direction_mode["saccade direction"]
analysis["saccade frequency (hz)"] = sac_per_sec_mean["saccade frequency (hz)"]
analysis["most common sequence"] = common_seq["sequence"]
analysis["sequence count"] = common_seq["count"]
analysis.to_excel(os.path.join(output_folder_path, "analysis.xlsx"))

# Overall averages for numerical data
numerical_mean = analysis[
    [
        "fixation duration (s)",
        "fixation frequency (hz)",
        "first fixation duration (s)",
        "last fixation duration (s)",
        "saccade duration(s)",
        "saccade distance (px)",
        "saccade frequency (hz)",
    ]
]
numerical_mean = numerical_mean.mean().to_frame()
numerical_mean.columns = ["overall average"]
numerical_mean.to_excel(
    os.path.join(output_folder_path, "overall numerical averages.xlsx")
)

# First time participants fixated on each feature
tagid = gaze_fixation.groupby(["participant_folder", "tag"])
first_tag = pd.concat([tagid.first()])
first_tag.reset_index(inplace=True)
first_tag = first_tag[["participant_folder", "tag", "time elapsed(s)"]]
first_tag.to_excel(
    os.path.join(output_folder_path, "first fixation on each feature.xlsx")
)

# Visualizations
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
plt.savefig(os.path.join(output_plots_folder_path, "time spent on each feature.png"))
plt.show()

# Frequency of a tag being the first fixation
first_fix_plot = analysis["first fixation"].value_counts().to_frame().reset_index()
plt.barh(first_fix_plot["first fixation"], first_fix_plot["count"])
plt.xlabel("Count")
plt.ylabel("Feature")
plt.title("No. of Times a Feature was the First Fixation")
plt.savefig(os.path.join(output_plots_folder_path, "first fixation features.png"))
plt.show()

# Frequency of a tag being the last fixation
last_fix_plot = analysis["last fixation"].value_counts().to_frame().reset_index()
plt.barh(last_fix_plot["last fixation"], last_fix_plot["count"])
plt.xlabel("Count")
plt.ylabel("Feature")
plt.title("No. of Times a Feature was the Last Fixation")
plt.savefig(os.path.join(output_plots_folder_path, "last fixation features.png"))
plt.show()

# Demographics
if env_var.DEMOGRAPHICS:
    demographic = pd.read_excel(os.path.join(data_folder_path, "demographic.xlsx"))

    # Add demographic data to analysis dataframe
    analysis = pd.merge(
        analysis,
        demographic,
        left_on="participant_folder",
        right_on="participant_folder",
        how="left",
    )

    if "age" in analysis.columns:
        analysis.sort_values("age", inplace=True)
        analysis["age group"] = pd.cut(
            analysis["age"],
            bins=env_var.AGE_GROUP_BINS,
            right=True,
            precision=0,
            include_lowest=True,
        )

    # Age
    if "age group" in analysis.columns:
        fix_dur_age = analysis.groupby("age group")["fixation duration (s)"].mean()
        fix_per_sec_age = analysis.groupby("age group")[
            "fixation frequency (hz)"
        ].mean()
        first_fix_dur_age = analysis.groupby("age group")[
            "first fixation duration (s)"
        ].mean()
        last_fix_dur_age = analysis.groupby("age group")[
            "last fixation duration (s)"
        ].mean()
        sac_dur_age = analysis.groupby("age group")["saccade duration(s)"].mean()
        sac_dist_age = analysis.groupby("age group")["saccade distance (px)"].mean()
        sac_per_sec_age = analysis.groupby("age group")["saccade frequency (hz)"].mean()

        age_data = fix_dur_age.to_frame()
        age_data["fixation frequency (hz)"] = fix_per_sec_age
        age_data["first fixation duration (s)"] = first_fix_dur_age
        age_data["last fixation duration (s)"] = last_fix_dur_age
        age_data["saccade duration (s)"] = sac_dur_age
        age_data["saccade distance (px)"] = sac_dist_age
        age_data["saccade frequency (hz)"] = sac_per_sec_age
        age_data.to_excel(
            os.path.join(output_folder_path, "numerical data by age.xlsx")
        )

    # Education
    if "education" in analysis.columns:
        fix_dur_edu = analysis.groupby("education")["fixation duration (s)"].mean()
        fix_per_sec_edu = analysis.groupby("education")[
            "fixation frequency (hz)"
        ].mean()
        first_fix_dur_edu = analysis.groupby("education")[
            "first fixation duration (s)"
        ].mean()
        last_fix_dur_edu = analysis.groupby("education")[
            "last fixation duration (s)"
        ].mean()
        sac_dur_edu = analysis.groupby("education")["saccade duration(s)"].mean()
        sac_dist_edu = analysis.groupby("education")["saccade distance (px)"].mean()
        sac_per_sec_edu = analysis.groupby("education")["saccade frequency (hz)"].mean()

        education_data = fix_dur_edu.to_frame()
        education_data["fixation frequency (hz)"] = fix_per_sec_edu
        education_data["first fixation duration (s)"] = first_fix_dur_edu
        education_data["last fixation duration (s)"] = last_fix_dur_edu
        education_data["saccade duration (s)"] = sac_dur_edu
        education_data["saccade distance (px)"] = sac_dist_edu
        education_data["saccade frequency (hz)"] = sac_per_sec_edu
        education_data.to_excel(
            os.path.join(output_folder_path, "numerical data by eduation.xlsx")
        )

    # Gender
    if "gender" in analysis.columns:
        fix_dur_gender = analysis.groupby("gender")["fixation duration (s)"].mean()
        fix_per_sec_gender = analysis.groupby("gender")[
            "fixation frequency (hz)"
        ].mean()
        first_fix_dur_gender = analysis.groupby("gender")[
            "first fixation duration (s)"
        ].mean()
        last_fix_dur_gender = analysis.groupby("gender")[
            "last fixation duration (s)"
        ].mean()
        sac_dur_gender = analysis.groupby("gender")["saccade duration(s)"].mean()
        sac_dist_gender = analysis.groupby("gender")["saccade distance (px)"].mean()
        sac_per_sec_gender = analysis.groupby("gender")["saccade frequency (hz)"].mean()

        gender_data = fix_dur_gender.to_frame()
        gender_data["fixation frequency (hz)"] = fix_per_sec_gender
        gender_data["first fixation duration (s)"] = first_fix_dur_gender
        gender_data["last fixation duration (s)"] = last_fix_dur_gender
        gender_data["saccade duration (s)"] = sac_dur_gender
        gender_data["saccade distance (px)"] = sac_dist_gender
        gender_data["saccade frequency (hz)"] = sac_per_sec_gender
        gender_data.to_excel(
            os.path.join(output_folder_path, "numerical data by gender.xlsx")
        )

    # Demographic visualizations (boxplots)
    vars_list = [
        ["fixation duration (s)", "age group"],
        ["fixation duration (s)", "education"],
        ["fixation duration (s)", "gender"],
        ["first fixation duration (s)", "age group"],
        ["first fixation duration (s)", "education"],
        ["first fixation duration (s)", "gender"],
        ["last fixation duration (s)", "age group"],
        ["last fixation duration (s)", "education"],
        ["last fixation duration (s)", "gender"],
        ["fixation frequency (hz)", "age group"],
        ["fixation frequency (hz)", "education"],
        ["fixation frequency (hz)", "gender"],
        ["saccade duration(s)", "age group"],
        ["saccade duration(s)", "education"],
        ["saccade duration(s)", "gender"],
        ["saccade distance (px)", "age group"],
        ["saccade distance (px)", "education"],
        ["saccade distance (px)", "gender"],
        ["saccade frequency (hz)", "age group"],
        ["saccade frequency (hz)", "education"],
        ["saccade frequency (hz)", "gender"],
    ]

    for vars in vars_list:
        try:
            analysis.boxplot(column=vars[0], by=vars[1])
            plt.xlabel(vars[0])
            plt.ylabel(vars[1])
            plt.xticks(rotation=90)
            plt.title(f"Plotting {vars[0]} by {vars[1]}")
            plt.suptitle("")
            path = os.path.join(output_plots_folder_path, f"{vars[0]}_{vars[1]}.png")
            plt.savefig(path)
        except:
            # when they lack a certain demographic variable
            print(f"{vars} not found in demographic data")
            continue

    if "gender" in analysis.columns:
        vars_list_group = [
            ["fixation duration (s)", "age group"],
            ["fixation duration (s)", "education"],
            ["first fixation duration (s)", "age group"],
            ["first fixation duration (s)", "education"],
            ["last fixation duration (s)", "age group"],
            ["last fixation duration (s)", "education"],
            ["fixation frequency (hz)", "age group"],
            ["fixation frequency (hz)", "education"],
            ["saccade duration(s)", "age group"],
            ["saccade duration(s)", "education"],
            ["saccade distance (px)", "age group"],
            ["saccade distance (px)", "education"],
            ["saccade frequency (hz)", "age group"],
            ["saccade frequency (hz)", "education"],
        ]

        for vars in vars_list_group:
            try:
                sns.boxplot(
                    x=analysis[f"{vars[1]}"],
                    y=analysis[f"{vars[0]}"],
                    hue=analysis["gender"],
                )
                plt.xticks(rotation=90)
                plt.title(f"Plotting {vars[0]} by {vars[1]} (grouped by gender)")
                path = os.path.join(
                    output_plots_folder_path, f"{vars[0]}_{vars[1]} by gender.png"
                )
                plt.show()
                plt.savefig(path)
            except:
                # when they lack a certain demographic variable
                print(f"{vars} not found in demographic data")
                continue
