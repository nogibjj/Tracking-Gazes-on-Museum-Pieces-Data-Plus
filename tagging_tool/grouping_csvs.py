"""This script combines all the csvs into one big csv for a given art piece"""

import pandas as pd
import os
import datetime as dt
import glob
import sys

path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, path)

# from config.config import *
from config import *
from heatmap.functions import create_directory

# print(" Made it past imports")
# Set env variables based on config file
try:
    env = sys.argv[1]
    env_var = eval(env + "_config")
except:
    print("Enter valid env variable. Refer to classes in the config.py file")
    sys.exit()

create_directory(env_var.TEMP_OUTPUT_DIR)


# A strong assumption is that
# we have all the participant folders
# within one folder
# This script cannot account for user error
# of having duplicate participant folders


participant_repository = None
name_of_art_piece = None


ROOT_PATH = env_var.ROOT_PATH
ART_PIECE = env_var.ART_PIECE  # a list


participant_paths_folders = []


for folder in os.listdir(ROOT_PATH):
    try:
        new_path = os.path.join(ROOT_PATH, folder)
        os.listdir(new_path)
        # Vulci Fixes Start Here
        # for special_file in os.listdir(new_path):
        #     try:
        #         further_path = os.path.join(new_path, special_file)
        #         os.listdir(further_path)
        #         print(f"{special_file} is a new data directory")
        #         participant_paths_folders.append(further_path)
        #     except:
        #         print(f"{special_file} is a file")
        #         continue
        # Vulci Fixes End Here
        print(f"Running for folder -- {folder}")
        print(f"{folder} is a directory")
        participant_paths_folders.append(new_path)

    except:
        print(f"{folder} is a file")
        continue

checkpoint = False
last_folder = None
participant_paths_folders = sorted(participant_paths_folders)
num_folders = len(participant_paths_folders)

if checkpoint:
    index_change = participant_paths_folders.index(last_folder)

    participant_paths_folders = participant_paths_folders[index_change + 1 :]

participant_list = []
vts_kids = []

# fixing news issue - Warning - Temporary fix
for folder in participant_paths_folders:
    files = os.listdir(folder)
    participant_id = folder.split(os.sep)[-1]
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
        if "vts" in folder.lower():
            vts_kids.append(participant_id)

# Concatenate all the tagged csvs into one big csv
# after preprocessing their timestamps
print(f"Number of folders found: {len(participant_paths_folders)}")
print(participant_paths_folders)
# assert len(participant_paths_folders) == 36
print(f"Number of participants found: {len(participant_list)}")
# assert len(participant_list) == 36
target_csv = pd.DataFrame()

ideal_rows = 0
participant_count = 0
for folder in participant_paths_folders:
    files = os.listdir(folder)
    participant_id = folder.split(os.sep)[-1]

    if "final_gaze_tagged.csv" in files:
        # if "gaze.csv" in files:
        print(f"file found in Processing {folder}")
        participant_count += 1
        gaze_csv_path = os.path.join(folder, "final_gaze_tagged.csv")
        # gaze_csv_path = os.path.join(folder, "gaze.csv")
        print(gaze_csv_path)
        # For now, we will keep the timestamp_corrector step
        # It might be redundant, but it is a good sanity check
        try:
            gaze_csv = pd.read_csv(gaze_csv_path)
        except:
            print(f"No 'final_gaze_tagged.csv' found in {folder}")
            continue
        gaze_csv["participant_folder"] = folder.split(os.sep)[-1]
        if len(ART_PIECE) > 1:
            gaze_csv["art_piece"] = ART_PIECE[0]  # change this line

        else:  # It is one piece only
            gaze_csv["art_piece"] = ART_PIECE[0]

        gaze_csv["participant_id"] = participant_count
        if participant_id in vts_kids:
            gaze_csv["vts_bool"] = True
        else:
            gaze_csv["vts_bool"] = False
        ideal_rows += gaze_csv.shape[0]
        target_csv = pd.concat([target_csv, gaze_csv], axis=0)
        del gaze_csv

try:
    target_csv.drop(columns=["Unnamed: 0"], inplace=True)

except:
    print("No Unnamed column found")
    pass

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

# change to all gaze for final implementation
target_csv.to_csv("data/all_gaze_Vulci_Final.csv", index=False, compression="gzip")
print("all_gaze_Vulci_Final.csv created")
