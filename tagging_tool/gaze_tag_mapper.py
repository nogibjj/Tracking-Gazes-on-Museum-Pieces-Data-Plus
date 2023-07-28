"""This is a script to apply tags to the gaze coordinates found in each participant's gaze.csv. 

The gaze point that falls next to the nearest box's center point will acquire the
tag associated with that box. 

A new csv file will be generated with the tag column appended to the end of the
gaze.csv file.

Author: Eric Rios-Soderman

"""
import pandas as pd
import os
import datetime as dt
import sys
from tag_event_functions import (
    gaze_tagger,
    ref_coordinate_processing,
)

path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, path)

from config.config import *

# print(" Made it past imports")
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
    files = os.listdir(os.path.join(output_folder_path, folder))
    participant_id = folder.split(os.sep)[-1]  # get the participant id
    # print(files)
    if index != 32:
        continue
    file_found = False
    for file in files:
        if "updated_gaze" in file and ".csv" in file:  # add participant as condition
            updated_gaze_path = os.path.join(data_folder_path, folder, file)
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
    # current_time = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    participant_output_path = os.path.join(output_folder_path, folder)
    # participant_reference_gaze_csv.to_csv(participant_output_path, f"final_gaze_tagged_{participant_id}_{current_time}.csv")
    participant_reference_gaze_csv.to_csv(
        os.path.join(participant_output_path, "final_gaze_tagged.csv")
    )

    del participant_reference_gaze_csv

    print("Done for folder -- ", folder)
