import sys
import os

path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# prepend parent directory to the system path:
sys.path.insert(0, path)

from collections import defaultdict
import glob
import pandas as pd
import numpy as np
import cv2
from heatmap.functions import (
    get_closest_individual_gaze_object,
    get_closest_reference_pixel,
    normalize_heatmap_dict,
    draw_heatmap_on_ref_img,
    create_directory,
    resample_gaze,
    save_outputs,
)
from helper_functions.timestamp_helper import convert_timestamp_ns_to_ms
from config import *
import datetime as dt

# # Set env variables based on config file
# try:
#     env = sys.argv[1]
#     env_var = eval(env + "_config")
# except:
#     print("Enter valid env variable. Refer to classes in the config.py file")
#     sys.exit()

# create_directory(env_var.TEMP_OUTPUT_DIR)


# ROOT_PATH = env_var.ROOT_PATH
ROOT_PATH = r"C:\Users\ericr\Desktop\Data + Plus\eye tracking data from the museum in Rome (Pupil Invisible)"
# MEMBER_FLAG = env

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

gazes_paths = []
gazes_reference_images = []
gazes_participants = []
gazes_videos = []
full_gaze = None
for folder in participant_paths_folders:
    files = os.listdir(folder)
    print(folder)
    participant_id = folder.split(os.sep)[-1]
    for file in files:
        if "gaze.csv" in file:
            gazes_paths.append(os.path.join(folder, file))
            gazes_participants.append(participant_id)
            if full_gaze is None:
                full_gaze = pd.read_csv(os.path.join(folder, file))

            else:
                full_gaze = pd.concat(
                    [full_gaze, pd.read_csv(os.path.join(folder, file))],
                    ignore_index=True,
                )

        if "reference_image" in file:
            gazes_reference_images.append(os.path.join(folder, file))

        if ".mp4" in file:
            gazes_videos.append(os.path.join(folder, file))

n = 14
trial_file = pd.read_csv(gazes_paths[n])
trial_participant = gazes_participants[n]
trial_reference_image = gazes_reference_images[n]
trial_video = gazes_videos[n]


# Address choosing a master reference image

# Address image shifting

cap = cv2.VideoCapture(trial_video)
gaze = trial_file

gaze["ts"] = gaze["timestamp [ns]"].apply(
    lambda x: dt.datetime.fromtimestamp(x / 1000000000)
)
baseline = gaze["ts"][0]
gaze["increment_marker"] = gaze["ts"] - baseline
gaze["seconds_id"] = gaze["increment_marker"].apply(lambda x: x.seconds) + 1
gaze_grouped_by_seconds = gaze.groupby("seconds_id")[
    ["gaze x [px]", "gaze y [px]"]
].mean()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fps = int(cap.get(cv2.CAP_PROP_FPS))
fps = int(cap.get(cv2.CAP_PROP_FPS))  # or 30
# fps = 30
fourcc = cv2.VideoWriter_fourcc(*"h264")  # mp4v
writer = cv2.VideoWriter("output.mp4", fourcc, fps, (frame_width, frame_height))

i = 1
frame_counter = 0
while True:
    # `success` is a boolean and `frame` contains the next video frame
    success, frame = cap.read()
    print(f"the previous frame is {i}, success is {success}")
    if success:
        frame_counter += 1
        if frame_counter > 30:
            i += 1  # moving the group by one increment
            frame_counter = 1

        try:
            x = gaze_grouped_by_seconds.iloc[i, 0]
            y = gaze_grouped_by_seconds.iloc[i, 1]
            x = int(x)
            y = int(y)

        except:
            # leftover frames, use final one
            x = gaze_grouped_by_seconds.iloc[-1, 0]
            y = gaze_grouped_by_seconds.iloc[-1, 1]
            x = int(x)
            y = int(y)

        cv2.rectangle(frame, (x - 30, y - 30), (x + 30, y + 30), (0, 255, 0), 1)
        writer.write(frame)
        cv2.imshow("output", frame)
        print(f"passed {i}")
        i += 1
        if cv2.waitKey(1) & 0xFF == ord("s"):
            break

    else:
        break

    # wait 20 milliseconds between frames and break the loop if the `q` key is pressed
    # if cv2.waitKey(20) == ord('q'):
    #     break


# we also need to close the video and destroy all Windows
cv2.destroyAllWindows()
cap.release()
writer.release()

# env_var.ROOT_PATH = ROOT_PATH
# for index, folder in enumerate(os.listdir(ROOT_PATH)):
for index, folder in enumerate(participant_paths_folders[:3]):
    # folder = os.path.join(ROOT_PATH, folder)
    folder_name = folder.split(os.sep)[-1]
    print("#" * 50)
    # print(f"Running for folder {index} -- {folder}")
    print(f"Running for folder {index} -- {folder_name}")
    pixel_heatmap = defaultdict(int)
    frame_no = 0

    # ToDo: Write a function to skip the first n frames based on percentage of grey/black color in the image
    name = folder.split(os.sep)[-1]

    csv_file = os.path.join(folder, "gaze.csv")

    video_file = os.path.join(folder, "*.mp4")
    video_file = glob.glob(video_file)[0]

    gaze_df = pd.read_csv(csv_file)
    gaze_df = convert_timestamp_ns_to_ms(gaze_df)
    if env_var.RESAMPLE:
        gaze_df = resample_gaze(gaze_df)

    updated_gaze = pd.DataFrame()
    cap = cv2.VideoCapture(video_file)

    while cap.isOpened():
        if frame_no % 1000 == 0:
            print(f"Processed {frame_no} frames")

        frame_no += 1
        frame_exists, curr_frame = cap.read()

        if frame_no < env_var.SKIP_FIRST_N_FRAMES:
            continue

        ##### Uncomment below if early stopping is required
        # if frame_no > SKIP_FIRST_N_FRAMES + RUN_FOR_FRAMES:
        #    break

        elif frame_no == env_var.SKIP_FIRST_N_FRAMES and frame_exists:
            first_frame = curr_frame

        elif frame_exists:
            gaze_object_crop, closest_row = get_closest_individual_gaze_object(
                cap, curr_frame, gaze_df, env_var.DETECT_BOUNDING_SIZE
            )

            if not gaze_object_crop.any() or closest_row.empty:
                continue

            ref_center = get_closest_reference_pixel(first_frame, gaze_object_crop)
            if ref_center == None:
                continue

            closest_row["ref_center_x"] = ref_center[0]
            closest_row["ref_center_y"] = ref_center[1]
            updated_gaze = pd.concat([updated_gaze, closest_row])
            pixel_heatmap[ref_center] += 1

            #####  Below code is just for plotting the centre of the images
            # _x = int(closest_row['gaze x [px]'].iloc[0])
            # _y = int(closest_row['gaze y [px]'].iloc[0])
            # pixel_heatmap[_x, _y] += 1

        else:
            break

    normalized_heatmap_dict = normalize_heatmap_dict(pixel_heatmap)
    final_img = draw_heatmap_on_ref_img(
        pixel_heatmap, np.copy(first_frame), env_var.DRAW_BOUNDING_SIZE
    )

    cap.release()
    save_outputs(
        env_var.ROOT_PATH,
        name,
        first_frame,
        env_var.DETECT_BOUNDING_SIZE,
        final_img,
        updated_gaze,
        env_var.TEMP_OUTPUT_DIR,
    )


for index, folder in enumerate(os.listdir(env_var.ROOT_PATH)):
    folder = os.path.join(env_var.ROOT_PATH, folder)
    print("#" * 50)
    print(f"Running for folder {index} -- {folder}")
    pixel_heatmap = defaultdict(int)
    frame_no = 0

    # ToDo: Write a function to skip the first n frames based on percentage of grey/black color in the image
    name = folder.split(os.sep)[-1]

    csv_file = os.path.join(folder, "gaze.csv")
    video_file = os.path.join(folder, "*.mp4")
    video_file = glob.glob(video_file)[0]

    gaze_df = pd.read_csv(csv_file)
    gaze_df = convert_timestamp_ns_to_ms(gaze_df)
    if env_var.RESAMPLE:
        gaze_df = resample_gaze(gaze_df)

    updated_gaze = pd.DataFrame()
    cap = cv2.VideoCapture(video_file)

    while cap.isOpened():
        if frame_no % 1000 == 0:
            print(f"Processed {frame_no} frames")

        frame_no += 1
        frame_exists, curr_frame = cap.read()

        if frame_no < env_var.SKIP_FIRST_N_FRAMES:
            continue

        ##### Uncomment below if early stopping is required
        # if frame_no > SKIP_FIRST_N_FRAMES + RUN_FOR_FRAMES:
        #    break

        elif frame_no == env_var.SKIP_FIRST_N_FRAMES and frame_exists:
            first_frame = curr_frame

        elif frame_exists:
            gaze_object_crop, closest_row = get_closest_individual_gaze_object(
                cap, curr_frame, gaze_df, env_var.DETECT_BOUNDING_SIZE
            )

            if not gaze_object_crop.any() or closest_row.empty:
                continue

            ref_center = get_closest_reference_pixel(first_frame, gaze_object_crop)
            if ref_center == None:
                continue

            closest_row["ref_center_x"] = ref_center[0]
            closest_row["ref_center_y"] = ref_center[1]
            updated_gaze = pd.concat([updated_gaze, closest_row])
            pixel_heatmap[ref_center] += 1

            #####  Below code is just for plotting the centre of the images
            # _x = int(closest_row['gaze x [px]'].iloc[0])
            # _y = int(closest_row['gaze y [px]'].iloc[0])
            # pixel_heatmap[_x, _y] += 1

        else:
            break

    normalized_heatmap_dict = normalize_heatmap_dict(pixel_heatmap)
    final_img = draw_heatmap_on_ref_img(
        pixel_heatmap, np.copy(first_frame), env_var.DRAW_BOUNDING_SIZE
    )

    cap.release()
    save_outputs(
        env_var.ROOT_PATH,
        name,
        first_frame,
        env_var.DETECT_BOUNDING_SIZE,
        final_img,
        updated_gaze,
        env_var.TEMP_OUTPUT_DIR,
    )
