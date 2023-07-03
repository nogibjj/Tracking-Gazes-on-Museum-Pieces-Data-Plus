"""
Author: Aditya John (aj391)
This is a script that ingests a csv of gazes (when looking at an object) and outputs a final image 
with a heatmap across the various pixels of interest. 
Spdates the gaze.csv file with new 'gaze' pixel locations that correspond to the reference image 
instead of the actual video. 

# ToDo:
- Add smoothing to the video file [pixel closeness? frame combination?]
- Figure out how to skip the first few grey frames in the video skip frame if % of greyness > thresh
- Change opacity of the bounding boxes? 
Ref: https://stackoverflow.com/questions/56472024/how-to-change-the-opacity-of-boxes-cv2-rectangle
- QC the outputs - April?
"""
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
from functions import (
    get_closest_individual_gaze_object,
    get_closest_reference_pixel,
    normalize_heatmap_dict,
    draw_heatmap_on_ref_img,
    create_directory,
    resample_gaze,
    save_outputs,
    reference_image_finder,
)
from helper_functions.timestamp_helper import convert_timestamp_ns_to_ms
from config import *

# Set env variables based on config file
try:
    env = sys.argv[1]
    env_var = eval(env + "_config")
except:
    print("Enter valid env variable. Refer to classes in the config.py file")
    sys.exit()

create_directory(env_var.TEMP_OUTPUT_DIR)

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
    print(
        f"""Starting to look for reference image for the video {video_file},
          from the folder {folder}"""
    )
    first_frame = reference_image_finder(video_file)
    print(
        f"""Found the reference image for the video {video_file}
          from the folder {folder}"""
    )
    cap = cv2.VideoCapture(video_file)

    while cap.isOpened():
        if frame_no % 1000 == 0:
            print(f"Processed {frame_no} frames")

        frame_no += 1
        frame_exists, curr_frame = cap.read()

        # if frame_no < env_var.SKIP_FIRST_N_FRAMES:
        #     continue

        ##### Uncomment below if early stopping is required
        # if frame_no > SKIP_FIRST_N_FRAMES + RUN_FOR_FRAMES:
        #    break

        # elif frame_no == env_var.SKIP_FIRST_N_FRAMES and frame_exists:
        #     first_frame = curr_frame

        if frame_exists:
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
