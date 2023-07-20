### Adding temporary solution for multiple folders
import os
from tag_event_functions import drawfunction
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import sys

path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, path)

from config.config import *
from heatmap.functions import create_directory

# print(" Made it past imports")
# Set env variables based on config file
try:
    env = sys.argv[1]
    env_var = eval(env + "_config")
except:
    print("Enter valid env variable. Refer to classes in the config.py file")
    sys.exit()

# to receive a copy of the tag coordinates
create_directory(env_var.TEMP_OUTPUT_DIR)

ROOT_PATH = env_var.ROOT_PATH

MEMBER_FLAG = env

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

participant_list = []

participant_paths_folders = sorted(participant_paths_folders)

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

participant_list = sorted(participant_list)

if checkpoint:
    with open("tagger_last_folder.txt", "r") as f:
        last_folder = f.read()

    index_change = participant_paths_folders.index(last_folder)

    participant_paths_folders = participant_paths_folders[index_change + 1 :]

for folder in participant_paths_folders:
    files = os.listdir(folder)
    print(folder)
    participant_id = folder.split(os.sep)[-1]
    feature_coordinates = []
    drawing = True
    flag = True
    list_finished = False
    if participant_id not in participant_list:
        print(f"Skipping folder -- {folder}")
        continue
    print(f"Running for folder -- {folder}")

    for single_file in files:
        if "reference_image" in single_file:
            file = os.path.join(folder, single_file)
            print(f"Running for file -- {single_file}")

    base_img = cv2.imread(file)

    param = [base_img, feature_coordinates]

    # base_img = cv2.imread("test4 image prompter.jpg")
    # img = cv2.cvtColor(base_img, cv2.COLOR_BGR2RGB)
    img = base_img
    reset_img = img.copy()
    # plt.imshow(img)

    # get the resolution of the image
    height, width, channels = img.shape
    print(f"width: {width}, height: {height}, channels: {channels}")

    cv2.namedWindow("image", flags=cv2.WINDOW_NORMAL)
    cv2.resizeWindow("image", width, height)

    cv2.setMouseCallback("image", drawfunction, param)

    while flag:
        cv2.imshow("image", img)
        key = cv2.waitKey(1)
        if key == ord("0"):
            break

        elif key == ord("5"):
            cv2.destroyAllWindows()
            img = reset_img.copy()
            print("You have reset the image")
            cv2.namedWindow("image", flags=cv2.WINDOW_NORMAL)
            cv2.resizeWindow("image", width, height)
            cv2.setMouseCallback("image", drawfunction, param)
            # cv2.imshow("image", img)

        elif key == ord("9"):
            flag = False
            print("You have finished tagging")
        # cv2.imshow("image", img)

    cv2.destroyAllWindows()

    coordinates_df = pd.DataFrame(
        feature_coordinates,
        columns=[
            "name",
            "(x1,y1)",
            "(x2,y2)",
            "(x3,y3)",
            "(x4,y4)",
            "(center_x,center_y)",
        ],
    )

    assert [i[0] for i in coordinates_df["(x1,y1)"]] == [
        i[0] for i in coordinates_df["(x3,y3)"]
    ]
    assert [i[0] for i in coordinates_df["(x2,y2)"]] == [
        i[0] for i in coordinates_df["(x4,y4)"]
    ]
    assert [i[1] for i in coordinates_df["(x1,y1)"]] == [
        i[1] for i in coordinates_df["(x4,y4)"]
    ]
    assert [i[1] for i in coordinates_df["(x2,y2)"]] == [
        i[1] for i in coordinates_df["(x3,y3)"]
    ]

    print("assertions passed")
    print(coordinates_df.head())

    current_time = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    print(f"Saving the csv to {os.path.join(folder, f'tags.csv')}")
    coordinates_df.to_csv(
        os.path.join(folder, f"tags_coordinates_{current_time}.csv"), index=False
    )
    coordinates_df.to_csv(
        os.path.join(env_var.TEMP_OUTPUT_DIR, f"tags_coordinates_{current_time}.csv"),
        index=False,
    )
    # market basket analysis
    # https://pbpython.com/market-basket-analysis.html

    last_folder = folder

    with open(f"tagger_last_folder.txt", "w") as f:
        f.write(last_folder)
        checkpoint = True
        f.write

    print(f"Finished generating tags for  {participant_id}")

    if folder == participant_paths_folders[-1]:
        print("Finished generating tags for all participants")
        os.remove("tagger_last_folder.txt")
