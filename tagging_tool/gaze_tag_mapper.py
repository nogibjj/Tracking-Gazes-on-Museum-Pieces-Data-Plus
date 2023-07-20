import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import copy
import gc
import math
import os
from tag_event_functions import drawfunction
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import re
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


def ref_coordinate_processing(gaze_reference_df):
    gaze_reference_df["ref_coordinates"] = pd.Series(
        zip(gaze_reference_df["ref_center_x"], gaze_reference_df["ref_center_y"])
    )

    x_coordinates_from_tuple = [
        i[0] for i in gaze_reference_df["ref_coordinates"].values.tolist()
    ]

    y_coordinates_from_tuple = [
        i[1] for i in gaze_reference_df["ref_coordinates"].values.tolist()
    ]

    assert gaze_reference_df["ref_center_x"].values.tolist() == x_coordinates_from_tuple

    assert gaze_reference_df["ref_center_y"].values.tolist() == y_coordinates_from_tuple

    return gaze_reference_df


def coordinate_parser(tuple_string):
    """Parse a string tuple into a list of integers.

    The tuple must be composed of coordinates.

    Example input: '(1,2)'

    Example output: [1,2]"""

    import re

    # remove the parentheses

    parsed_tuple = re.findall(pattern="\((\d+), (\d+)\)", string=tuple_string)[0]

    formatted_tuple = [int(i) for i in parsed_tuple]

    return formatted_tuple


def gaze_tagger(gaze_reference_df_obs, tags_df):
    """Tag the observation gaze with the feature"""

    # tag_list_from_df = user_tag_coordinates.values.tolist()

    # observation from that updated gaze csv
    # obtained and parsed from ref_coordinates

    # obs_x = gaze_reference_df_obs["ref_coordinates"][0]
    # obs_y = gaze_reference_df_obs["ref_coordinates"][1]
    # print(gaze_reference_df_obs, gaze_reference_df_obs[0])
    obs_x, obs_y = gaze_reference_df_obs[0], gaze_reference_df_obs[1]

    # extracting the features for the following loop operation

    features = [i for i in tags_df["name"].unique()]

    name = "noise"

    smallest_center_x, smallest_center_y = None, None

    distance_from_center = None

    for feature in features:
        # the coordinates here are the
        # points of the rectangle encapsulated
        # by the user's bounding in the
        # tagging_event.py script

        # upper left corner
        p1 = user_tag_coordinates.loc[
            user_tag_coordinates["name"] == feature, "(x1,y1)"
        ].values.tolist()
        x1, y1 = coordinate_parser(p1[0])

        # lower right corner
        p2 = user_tag_coordinates.loc[
            user_tag_coordinates["name"] == feature, "(x2,y2)"
        ].values.tolist()
        x2, y2 = coordinate_parser(p2[0])

        # lower left corner
        p3 = user_tag_coordinates.loc[
            user_tag_coordinates["name"] == feature, "(x3,y3)"
        ].values.tolist()
        x3, y3 = coordinate_parser(p3[0])

        # upper right corner
        p4 = user_tag_coordinates.loc[
            user_tag_coordinates["name"] == feature, "(x4,y4)"
        ].values.tolist()
        x4, y4 = coordinate_parser(p4[0])

        center = user_tag_coordinates.loc[
            user_tag_coordinates["name"] == feature, "(center_x,center_y)"
        ].values.tolist()
        center_x, center_y = coordinate_parser(center[0])

        # the if statements to check if the observation gaze
        # is within the bounds of the rectangle

        if (
            (obs_x >= x1 and obs_y <= y1)
            and (obs_x <= x2 and obs_y >= y2)
            and (obs_x >= x3 and obs_y >= y3)
            and (obs_x <= x4 and obs_y <= y4)
        ):
            if (
                name != "noise"
                and smallest_center_x != None
                and smallest_center_y != None
                and distance_from_center != None
            ):
                # if the observation gaze is within the bounds
                # of two or more rectangles, the one with the
                # smallest center is chosen
                # print("current gaze", obs_x, obs_y)
                # print("current name : ", name)
                # print("current center : ", smallest_center_x, smallest_center_y)
                # print("current distance : ", distance_from_center)
                # print("competing name : ", feature)
                # print("competing center : ", center_x, center_y)
                # print(
                #     "competing distance : ",
                #     math.dist((obs_x, obs_y), (center_x, center_y)),
                # )
                if (
                    math.dist((obs_x, obs_y), (center_x, center_y))
                    < distance_from_center
                ):
                    name = copy.deepcopy(feature)

                    smallest_center_x, smallest_center_y = copy.deepcopy(
                        center_x
                    ), copy.deepcopy(center_y)

                    # print("new name : ", name)
                    # print("new center : ", smallest_center_x, smallest_center_y)

            else:
                name = copy.deepcopy(feature)

                smallest_center_x, smallest_center_y = copy.deepcopy(
                    center_x
                ), copy.deepcopy(center_y)

                distance_from_center = math.dist((obs_x, obs_y), (center_x, center_y))

    gc.collect()

    return name


ROOT_PATH = env_var.ROOT_PATH

participant_paths_folders = []

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


for folder in participant_paths_folders:
    print(f"Starting {folder}")
    files = os.listdir(folder)
    participant_id = folder.split(os.sep)[-1]  # fix for future reference
    # print(files)
    if participant_id not in participant_list:
        print(f"Skipping folder -- {folder}")
        continue

    most_recent_tag_file = None
    most_recent_date = None
    for file in files:
        if "tags_coordinates" in file and ".csv" in file:
            extracted_date = re.findall(
                "tags_coordinates_(.+)\.csv", "tags_coordinates_2023-06-20_11-41-42.csv"
            )[0]
            parsed_date = dt.datetime.strptime(extracted_date, "%Y-%m-%d_%H-%M-%S")

            if most_recent_date is None and most_recent_tag_file is None:
                most_recent_tag_file = file
                most_recent_date = parsed_date

            elif parsed_date > most_recent_date:
                most_recent_tag_file = file
                most_recent_date = parsed_date

        elif "updated_gaze" in file and ".csv" in file:  # add participant as condition
            participant_reference_gaze_csv = pd.read_csv(os.path.join(folder, file))
            print(f"Participant Csv found : {os.path.join(folder, file)}")
            print("*" * 50)

    tag_file_path = os.path.join(folder, most_recent_tag_file)
    print(tag_file_path)
    user_tag_coordinates = pd.read_csv(tag_file_path)

    # assuming a group reference image
    # or a participant reference image
    # is received, this code should work agnostically

    # test
    # participant_reference_gaze_csv = pd.read_csv("test reference gaze updated.csv")
    # participant_reference_gaze_csv = pd.read_csv("gaze_csv_tag_exp_sb.csv")

    try:
        participant_reference_gaze_csv = ref_coordinate_processing(
            participant_reference_gaze_csv
        )

    except:
        print(f"No Updated Gaze Csv found for folder : {folder}")
        continue

    participant_reference_gaze_csv["tag"] = participant_reference_gaze_csv[
        "ref_coordinates"
    ].apply(lambda x: gaze_tagger(x, user_tag_coordinates))
    current_time = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    participant_reference_gaze_csv.to_csv(
        os.path.join(
            env_var.TEMP_OUTPUT_DIR,
            f"final_gaze_tagged_{participant_id}_{current_time}.csv",
        )
    )
    participant_reference_gaze_csv.to_csv(os.path.join(folder, "final_gaze_tagged.csv"))

    del participant_reference_gaze_csv

    print("Done for folder -- ", folder)
