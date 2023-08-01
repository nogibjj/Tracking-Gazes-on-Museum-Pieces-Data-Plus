"""This is a script to apply tags to the gaze coordinates found in each participant's gaze.csv. 

The gaze point that falls next to the nearest box's center point will acquire the
tag associated with that box. 

A new csv file will be generated with the tag column appended to the end of the
gaze.csv file.

The second half of the script combines all the participants' tagged csvs into one final csv.

It is specifically reading the final_tagged_csvs, the gaze files that are tagged
with the names of the features that people were observing.

Author: Eric Rios-Soderman

"""
import pandas as pd
import os
import datetime as dt
import sys
from map_functions import (
    gaze_tagger,
    ref_coordinate_processing,
)

path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, path)

from config.config import *

# Set env variables based on config file
try:
    env = sys.argv[1]
    env_var = eval(env + "_config")
except:
    print("Enter valid env variable. Refer to classes in the config.py file")
    sys.exit()

# the path will be the same, regardless if supplied or generated
data_folder_path = os.path.join(
    env_var.ROOT_PATH, env_var.ART_PIECE
)  # all the participant folders are here

output_folder_path = os.path.join(env_var.OUTPUT_PATH, env_var.ART_PIECE)

tag_file_path = os.path.join(output_folder_path, "tags_coordinates.csv")
user_tag_coordinates = pd.read_csv(tag_file_path)

if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

for index, folder in enumerate(sorted(os.listdir(output_folder_path))):
    print(f"Starting folder {index} : {folder}")
    try:
        files = os.listdir(os.path.join(output_folder_path, folder))
    except:
        print(f"{folder} is not a folder")
        continue
    participant_id = folder.split(os.sep)[-1]  # get the participant id

    file_found = False
    for file in files:
        if "updated_gaze" in file and ".csv" in file:
            updated_gaze_path = os.path.join(output_folder_path, folder, file)
            participant_reference_gaze_csv = pd.read_csv(updated_gaze_path)
            file_found = True

    if not file_found:
        print(f"No Updated Gaze Csv found for folder : {folder}")
        continue

    participant_reference_gaze_csv = ref_coordinate_processing(
        participant_reference_gaze_csv
    )

    participant_reference_gaze_csv["tag"] = participant_reference_gaze_csv[
        "ref_coordinates"
    ].apply(lambda x: gaze_tagger(x, user_tag_coordinates))
    participant_output_path = os.path.join(output_folder_path, folder)
    participant_reference_gaze_csv.to_csv(
        os.path.join(participant_output_path, "final_gaze_tagged.csv")
    )

    del participant_reference_gaze_csv

    print("Done for folder -- ", folder)

# the final csv which will house the concatenated rows
target_csv = pd.DataFrame()

participant_list = []
no_gaze_csv = []

for folder in sorted(os.listdir(data_folder_path)):
    try:
        os.listdir(os.path.join(data_folder_path, folder))
        participant_list.append(folder)

    except:
        print(f"{folder} is not a folder")
        continue

ideal_rows = 0
participant_count = 0
for folder in sorted(os.listdir(output_folder_path)):
    try:
        files = os.listdir(os.path.join(output_folder_path, folder))
    except:
        print(f"{folder} is not a folder")
        continue
    participant_id = folder.split(os.sep)[-1]

    gaze_csv_path = os.path.join(
        os.path.join(output_folder_path, folder), "final_gaze_tagged.csv"
    )

    try:
        gaze_csv = pd.read_csv(gaze_csv_path)
    except:
        print(f"No 'final_gaze_tagged.csv' found in {folder}")
        no_gaze_csv.append(folder)
        continue

    print(f"file found in Processing {folder}")
    participant_count += 1

    gaze_csv["participant_folder"] = folder.split(os.sep)[-1]

    # this vts bool is a distinction made for our stakeholders
    # It is representative of surveys that were taken during the study,
    # which are only applicable to the participants that took them.
    if "vts" in participant_id:
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
    print(f"Number of folders found: {len(participant_list)}")
    print(f"Participants unable to be processed : {no_gaze_csv}")
    print(f"Participant count does not match number of folders")

    with open(
        os.path.join(output_folder_path, "Error Log - Folder Count.txt"), "w"
    ) as f:
        f.write(f"Number of participants found: {participant_count}")
        f.write(f"Number of folders found: {len(participant_list)}")
        f.write(f"Participant count does not match number of folders")
        f.write(f"Participants without gaze csvs : {no_gaze_csv}")

# change to all gaze for final implementation
target_csv.to_csv(
    os.path.join(output_folder_path, "all_gaze.csv"), index=False, compression="gzip"
)
print("all_gaze.csv created")