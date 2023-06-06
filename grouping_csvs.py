"""This script combines all the csvs into one big csv for a given art piece"""

import pandas as pd
import os
import datetime as dt


def timestamp_corrector(gaze_csv_path):
    """Process the unix timestamps
    and create seconds columns to facilitate
    generation of descriptive statistics"""

    gaze_copy = pd.read_csv(gaze_csv_path)
    gaze_copy["ts"] = gaze_copy["timestamp [ns]"].apply(
        lambda x: dt.datetime.fromtimestamp(x / 1000000000)
    )
    baseline = gaze_copy["ts"][0]
    gaze_copy["increment_marker"] = gaze_copy["ts"] - baseline
    gaze_copy["seconds_id"] = gaze_copy["increment_marker"].apply(lambda x: x.seconds)
    return gaze_copy


# A strong assumption is that
# we have all the participant folders
# within one folder
# This script cannot account for user error
# of having duplicate participant folders


participant_repository = None

with open("Paths.txt", "r") as f:
    for line in f.readlines():
        if "AFTER THE COLON" in line:
            participant_repository = line.split(":", maxsplit=1)[1].strip()

            print(f"participant_repository is {participant_repository}")

            break

participant_folders = os.listdir(participant_repository)
num_folders = len(participant_folders)

#######################

# Placeholder function for tagging mechanism

#######################

# Concatenate all the tagged csvs into one big csv
# after preprocessing their timestamps
participant_count = 0
for folder in participant_folders:
    files = os.listdir(os.path.join(participant_repository, folder))

    if "gaze.csv" in files:
        print(f"file found in Processing {folder}")
        participant_count += 1

    assert participant_count == len(num_folders)

print(f"Number of participants found: {participant_count}")
print(num_folders == participant_count)
