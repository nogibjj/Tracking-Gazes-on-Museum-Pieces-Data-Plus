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

target_csv = pd.DataFrame()


participant_count = 0
for folder in participant_folders:
    files = os.listdir(os.path.join(participant_repository, folder))

    if "gaze.csv" in files:
        print(f"file found in Processing {folder}")
        gaze_csv_path = os.path.join(participant_repository, folder, "gaze.csv")
        # For now, we will keep the timestamp_corrector step
        # It might be redundant, but it is a good sanity check
        gaze_csv = timestamp_corrector(gaze_csv_path)
        target_csv = pd.concat([target_csv, gaze_csv], axis=0)
        participant_count += 1

try:
    assert participant_count == num_folders

except AssertionError:
    print(f"Number of participants found: {participant_count}")
    print(f"Number of folders found: {num_folders}")
    print(f"Participant count does not match number of folders")

    with open("Error Log - Folder Count.txt", "w") as f:
        f.write(f"Number of participants found: {participant_count}")
        f.write(f"Number of folders found: {num_folders}")
        f.write(f"Participant count does not match number of folders")

target_csv.to_csv("all_gaze.csv", index=False, compression="gzip")
print("all_gaze.csv created")
