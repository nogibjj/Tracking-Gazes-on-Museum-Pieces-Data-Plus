"""This script combines all the csvs into one big csv for a given art piece"""

import pandas as pd
import os
import datetime as dt
import glob
from repository_finder import repository_details


def timestamp_corrector(gaze_csv_path, col_name="timestamp [ns]_for_grouping"):
    """Process the unix timestamps
    and create seconds columns to facilitate
    generation of descriptive statistics"""

    gaze_copy = pd.read_csv(gaze_csv_path)
    gaze_copy["ts"] = gaze_copy[col_name].apply(
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
name_of_art_piece = None

with open("Paths.txt", "r") as f:
    for line in f.readlines():
        if "AFTER THE COLON" in line and "PARTICIPANT" in line:
            participant_repository = line.split(":", maxsplit=1)[1].strip()

            print(f"participant_repository is {participant_repository}")

        elif "AFTER THE COLON" in line and "ART PIECE" in line:
            name_of_art_piece = line.split(":", maxsplit=1)[1].strip()

            break


ROOT_PATH, ART_PIECE = repository_details("Paths.txt")


participant_paths_folders = []

num_folders = len(participant_paths_folders)

for folder in os.listdir(ROOT_PATH):
    try:
        new_path = os.path.join(ROOT_PATH, folder)
        os.listdir(new_path)
        print(f"Running for folder -- {folder}")
        print(f"{folder} is a directory")
        participant_paths_folders.append(new_path)

    except:
        print(f"{folder} is a file")
        continue

checkpoint = False
last_folder = None
participant_paths_folders = sorted(participant_paths_folders)

if checkpoint:
    index_change = participant_paths_folders.index(last_folder)

    participant_paths_folders = participant_paths_folders[index_change + 1 :]

participant_list = []

# fixing news issue - Warning - Temporary fix
for folder in participant_paths_folders:
    files = os.listdir(folder)
    participant_id = folder.split("\\")[-1]
    if "new" in participant_id:
        print(f"Fixing participant id -- {participant_id}")
        temp = participant_id.replace("new", "")
        flip_flag = False
        for id in participant_list:
            if temp in id:
                print(f"Replacing {id} with {participant_id}")
                ind = participant_list.index(id)
                flip_flag = True
                participant_list.pop(ind)
                participant_list.append(participant_id)
        if not (flip_flag):
            participant_list.append(participant_id)
            flip_flag = False
    else:
        participant_list.append(participant_id)

# Concatenate all the tagged csvs into one big csv
# after preprocessing their timestamps

target_csv = pd.DataFrame()

ideal_rows = 0
participant_count = 0
for folder in participant_paths_folders:
    files = os.listdir(folder)

    if "final_gaze_tagged.csv" in files:
        print(f"file found in Processing {folder}")
        participant_count += 1
        gaze_csv_path = os.path.join(folder, "final_gaze_tagged.csv")
        # print(gaze_csv_path)
        # For now, we will keep the timestamp_corrector step
        # It might be redundant, but it is a good sanity check
        gaze_csv = timestamp_corrector(gaze_csv_path)
        gaze_csv["participant_folder"] = folder.split("\\")[-1]
        gaze_csv["art_piece"] = name_of_art_piece
        gaze_csv["participant_id"] = participant_count
        ideal_rows += gaze_csv.shape[0]
        target_csv = pd.concat([target_csv, gaze_csv], axis=0)

target_csv.drop(columns=["Unnamed: 0"], inplace=True)
assert ideal_rows == target_csv.shape[0]


try:
    assert participant_count == len(participant_list)

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
