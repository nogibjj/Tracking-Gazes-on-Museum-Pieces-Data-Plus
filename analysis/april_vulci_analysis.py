"""Descriptive statistics of the Vulci site"""

import pandas as pd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math


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
all_gaze = pd.read_csv("../data/all_gaze_Vulci.csv", compression="gzip")

all_gaze.reset_index(drop=True, inplace=True)
gaze_copy = all_gaze

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
gaze_fixed = gaze_copy[~gaze_copy["participant_folder"].isin(null_participants)]


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
gaze_saccades = gaze_saccades.drop("row_number", axis=1)
gaze_saccades.reset_index(inplace=True)

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
saccade_calc.reset_index(drop=True, inplace=True)

gaze_saccades = pd.merge(
    gaze_saccades, saccade_calc, left_on=["index"], right_on="index", how="left"
)
id2 = gaze_saccades.groupby(["participant_folder", "fixation id"])
rearrange = gaze_saccades[gaze_saccades["fixation id"].isnull()]
gaze_saccades = pd.concat([id2.tail(1)])
gaze_saccades = pd.concat([rearrange, gaze_saccades])
gaze_saccades = gaze_saccades.sort_values("index").reset_index(drop=True)
gaze_saccades.reset_index(drop=True, inplace=True)

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
saccade_time["time elapsed(s)"] = saccade_time.groupby("participant_folder")[
    "duration"
].cumsum()
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

fix_mean = fix_durations.groupby("participant_folder")["fixation duration(s)"].mean()
fix_mean = fix_mean.to_frame()
fix_mean = fix_mean.rename({"fixation duration(s)": "mean fix duration(s)"}, axis=1)

overall_mean = fix_mean["mean fix duration(s)"].mean()

# Fixations per second for each participant
fix_per_time = gaze_fixed[
    ["participant_folder", "fixation id", "duration"]
].reset_index(drop=True)
fix_per_time["time elapsed(s)"] = fix_per_time.groupby("participant_folder")[
    "duration"
].cumsum()

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


# Putting together cumulative analysis dataframe
analysis = fix_mean.reset_index()
analysis["fixation freq"] = fix_per_sec_mean["fixation freq"]
analysis["mean sac duration(s)"] = sac_dur_mean["saccade duration(s)"]
analysis["mean sac distance"] = sac_distance_mean["saccade distance"]
analysis["mode sac direction"] = direction_mode["saccade direction"]
analysis["saccade freq"] = sac_per_sec_mean["saccade freq"]

# VTS vs non-VTS
analysis_vts = pd.merge(
    analysis,
    gaze_fixed[["participant_folder", "vts_bool"]],
    left_on="participant_folder",
    right_on="participant_folder",
    how="left",
)
id3 = analysis_vts.groupby("participant_folder")
analysis_vts = pd.concat([id3.head(1)])
analysis_vts.reset_index(drop=True, inplace=True)
analysis_vts["vts_bool"] = analysis_vts["vts_bool"].map({False: "Non-VTS", True: "VTS"})
vts = analysis_vts[analysis_vts["vts_bool"] == True]
nonvts = analysis_vts[analysis_vts["vts_bool"] == False]

# Overall means
vts_analysis = analysis_vts.drop(["participant_folder", "mode sac direction"], axis=1)
analysis_mean = vts_analysis.groupby("vts_bool").mean().transpose()
analysis_mean["Overall Mean"] = (analysis_mean["Non-VTS"] + analysis_mean["VTS"]) / 2

# Make tables an excel
analysis_vts.to_excel("vulci-raw-metrics.xlsx")
analysis_mean.to_excel("overall-means.xlsx")

# Visualizations (boxplot)
plt.style.use("ggplot")

fix_dur_plot = analysis_vts.boxplot(column="mean fix duration(s)", by="vts_bool")
plt.xlabel("Group")
plt.ylabel("Duration(s)")
plt.title("Fixation Duration(s) in VTS vs Non-VTS Participants")
plt.suptitle("")
# plt.savefig("Mean Fixation Duration(s) Plot")
plt.show()

fix_freq_plot = analysis_vts.boxplot(column="fixation freq", by="vts_bool")
plt.xlabel("Group")
plt.ylabel("Frequency per second")
plt.title("Fixation Frequency in VTS vs Non-VTS Participants")
plt.suptitle("")
# plt.savefig("Mean Fixation Frequency Plot")
plt.show()

sac_dur_plot = analysis_vts.boxplot(column="mean sac duration(s)", by="vts_bool")
plt.xlabel("Group")
plt.ylabel("Duration(s)")
plt.title("Saccade Duration(s) in VTS vs Non-VTS Participants")
plt.suptitle("")
# plt.savefig("Mean Saccade Duration(s) Plot")
plt.show()

sac_dist_plot = analysis_vts.boxplot(column="mean sac distance", by="vts_bool")
plt.xlabel("Group")
plt.ylabel("Distance(px)")
plt.title("Saccade Distance in VTS vs Non-VTS Participants")
plt.suptitle("")
# plt.savefig("Mean Saccade Distance Plot")
plt.show()

sac_freq_plot = analysis_vts.boxplot(column="saccade freq", by="vts_bool")
plt.xlabel("Group")
plt.ylabel("Frequency per second")
plt.title("Saccade Frequency in VTS vs Non-VTS Participants")
plt.suptitle("")
# plt.savefig("Mean Saccade Frequency Plot")
plt.show()


"""vtslist = [vts['mean fix duration(s)'], vts['mean sac duration(s)']]
nonvtslist = [nonvts['mean fix duration(s)'], nonvts['mean sac duration(s)']]
 
# the list named ticks, summarizes or groups
# the summer and winter rainfall as low, mid
# and high
ticks = ['Fixations', 'Saccades']
 
# create a boxplot for two arrays separately,
# the position specifies the location of the
# particular box in the graph,
# this can be changed as per your wish. Use width
# to specify the width of the plot
vts_plot = plt.boxplot(vtslist,
                               positions=np.array(
    np.arange(len(vtslist)))*2.0-0.35,
                               widths=0.6, showfliers=False)
nonvts_plot = plt.boxplot(nonvtslist,
                               positions=np.array(
    np.arange(len(nonvtslist)))*2.0+0.35,
                               widths=0.6, showfliers=False)
 
# each plot returns a dictionary, use plt.setp()
# function to assign the color code
# for all properties of the box plot of particular group
# use the below function to set color for particular group,
# by iterating over all properties of the box plot
def define_box_properties(plot_name, color_code, label):
    for k, v in plot_name.items():
        plt.setp(plot_name.get(k), color=color_code)
         
    # use plot function to draw a small line to name the legend.
    plt.plot([], c=color_code, label=label)
    plt.legend()
 
 
# setting colors for each groups
define_box_properties(vts_plot, '#D7191C', 'VTS')
define_box_properties(nonvts_plot, '#2C7BB6', 'Non-VTS')
 
# set the x label values
plt.xticks(np.arange(0, len(ticks) * 2, 2), ticks)
 
# set the limit for x axis
plt.xlim(-2, len(ticks)*2)
 
# set the limit for y axis
plt.ylim(0, 0.8)
 
# set the title
plt.title('Duration(s) Comparisons for VTS vs Non-VTS Participants')
plt.show()
"""
