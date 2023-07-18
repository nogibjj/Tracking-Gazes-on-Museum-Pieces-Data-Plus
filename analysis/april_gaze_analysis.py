"""This script describe statistics of when/where people are looking"""

import pandas as pd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import math
import collections
from collections import Counter
import scipy

plt.style.use("ggplot")


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
all_gaze = pd.read_csv("../data/all_gaze_original.csv", compression="gzip")

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

# Figuring out the participants who don't have a fixation id
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


# Demographics
# Add demographic data to analysis dataframe
analysis = pd.merge(
    analysis,
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
analysis.drop("codice_eyetr_museo", axis=1, inplace=True)
analysis.sort_values("Age", inplace=True)
analysis["age group"] = pd.cut(
    analysis["Age"], bins=6, right=True, precision=0, include_lowest=True
)

analysis_men = analysis[analysis["sesso"] == "m"]
analysis_women = analysis[analysis["sesso"] == "f"]

# Fixation means with demographic data
fix_mean_age = analysis.groupby("age group")["mean fix duration(s)"].mean()
fix_mean_edu = analysis.groupby("Educational Qualification")[
    "mean fix duration(s)"
].mean()
fix_mean_gender = analysis.groupby("sesso")["mean fix duration(s)"].mean()

# Fixation frequency per second with demographic data
fix_per_sec_age = analysis.groupby("age group")["fixation freq"].mean()
fix_per_sec_edu = analysis.groupby("Educational Qualification")["fixation freq"].mean()
fix_per_sec_gender = analysis.groupby("sesso")["fixation freq"].mean()

# Fixation frequency boxplot
analysis.boxplot(column="fixation freq", by="age group")
plt.xlabel("Age Group")
plt.ylabel("Frequency per second")
plt.title("Fixation Frequency by Age")
plt.suptitle("")
plt.show()

# Fixation duration with demographic data
fix_dur_men = analysis_men[["participant_folder", "mean fix duration(s)", "age group"]]
fix_dur_men["age group"] = fix_dur_men["age group"].astype(str)

fix_dur_women = analysis_women[
    ["participant_folder", "mean fix duration(s)", "age group"]
]
fix_dur_women["age group"] = fix_dur_women["age group"].astype(str)

age1m = fix_dur_men[fix_dur_men["age group"] == "(16.0, 25.0]"]
age2m = fix_dur_men[fix_dur_men["age group"] == "(25.0, 32.0]"]
age3m = fix_dur_men[fix_dur_men["age group"] == "(32.0, 40.0]"]
age4m = fix_dur_men[fix_dur_men["age group"] == "(40.0, 48.0]"]

age1w = fix_dur_women[fix_dur_women["age group"] == "(16.0, 25.0]"]
age2w = fix_dur_women[fix_dur_women["age group"] == "(25.0, 32.0]"]
age3w = fix_dur_women[fix_dur_women["age group"] == "(32.0, 40.0]"]
age4w = fix_dur_women[fix_dur_women["age group"] == "(40.0, 48.0]"]
age5w = fix_dur_women[fix_dur_women["age group"] == "(55.0, 63.0]"]

# Duration t test
group1 = scipy.stats.ttest_ind(
    age1m["mean fix duration(s)"].dropna(), age1w["mean fix duration(s)"].dropna()
)
group2 = scipy.stats.ttest_ind(
    age2m["mean fix duration(s)"].dropna(), age2w["mean fix duration(s)"].dropna()
)
group3 = scipy.stats.ttest_ind(
    age3m["mean fix duration(s)"].dropna(), age3w["mean fix duration(s)"].dropna()
)
group4 = scipy.stats.ttest_ind(
    age4m["mean fix duration(s)"].dropna(), age4w["mean fix duration(s)"].dropna()
)

age_dur_t = [group1, group2, group3, group4]
age_dur_t = pd.DataFrame(age_dur_t)
age_dur_t.index = ["(16.0, 25.0]", "(25.0, 32.0]", "(32.0, 40.0]", "(40.0, 48.0]"]


def convert_pvalue_to_asterisks(pvalue):
    if pvalue <= 0.0001:
        return "****"
    elif pvalue <= 0.001:
        return "***"
    elif pvalue <= 0.01:
        return "**"
    elif pvalue <= 0.05:
        return "*"
    return "ns"


age_dur_asterisks = age_dur_t["pvalue"].apply(convert_pvalue_to_asterisks)

# Fixation duration errorbar plot
fix_dur_men_plot = (
    fix_dur_men.groupby("age group")["mean fix duration(s)"].mean().to_frame()
)
fix_dur_men_plot.dropna(inplace=True)
fix_dur_men_plot["sem"] = fix_dur_men.groupby("age group")["mean fix duration(s)"].sem()
fix_dur_men_plot["sem"].fillna(0, inplace=True)

fix_dur_women_plot = (
    fix_dur_women.groupby("age group")["mean fix duration(s)"].mean().to_frame()
)
fix_dur_women_plot.dropna(inplace=True)
fix_dur_women_plot["sem"] = fix_dur_women.groupby("age group")[
    "mean fix duration(s)"
].sem()
fix_dur_women_plot["sem"].fillna(0, inplace=True)

plt.errorbar(
    fix_dur_men_plot.index,
    fix_dur_men_plot["mean fix duration(s)"],
    yerr=fix_dur_men_plot["sem"],
    capsize=4,
    color="blue",
    label="men",
)

plt.errorbar(
    fix_dur_women_plot.index,
    fix_dur_women_plot["mean fix duration(s)"],
    yerr=fix_dur_women_plot["sem"],
    capsize=4,
    color="red",
    label="women",
)

"""plt.text(0.2, 0.62, "10", ha="center")
plt.text(1.1, 0.52, "2", ha="center")
plt.text(2, 0.45, "1", ha="center")
plt.text(3, 0.57, "1", ha="center")

plt.text(0, 0.43, "12", ha="center")
plt.text(1, 0.43, "7", ha="center")
plt.text(2, 0.345, "1", ha="center")
plt.text(3, 0.35, "2", ha="center")
plt.text(4, 0.57, "1", ha="center")

plt.text(0, 0.94, "ns", ha="center")
plt.text(1, 0.57, "ns", ha="center")
plt.text(2, 0.51, "ns", ha="center")
plt.text(3, 0.62, "*", ha="center")"""

plt.legend()
plt.xlabel("Age Group")
plt.ylabel("Fixation Duration (s)")
plt.title("Fixation Duration by Gender and Age")
plt.show()
