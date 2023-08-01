"""
Author: Aditya John (aj391), Eric Rios Soderman (ejr41) (Reference Image Implementation)
This is a script that ingests a csv of gazes (when looking at an object) and outputs a final image 
with a heatmap across the various pixels of interest. 
Updates the gaze.csv file with new 'gaze' pixel locations that correspond to the reference image 
instead of the actual video. 
"""
import sys
import os
import time

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
    normalize_heatmap_dict,
    draw_heatmap_on_ref_img,
    save_outputs,
    reference_gaze_point_mapper,
)
from create_reference_image.functions import is_single_color
import matplotlib.pyplot as plt
from helper_functions.timestamp_helper import convert_timestamp_ns_to_ms
from config.config import *

# Set env variables based on config file
try:
    env = sys.argv[1]
    env_var = eval(env + "_config")
except:
    print("Enter valid env variable. Refer to classes in the config.py file")
    sys.exit()

data_folder_path = os.path.join(env_var.ROOT_PATH, env_var.ART_PIECE)
output_folder_path = os.path.join(env_var.OUTPUT_PATH, env_var.ART_PIECE)

for index, folder in enumerate(os.listdir(data_folder_path)):
    print("#" * 50)
    print(f"Running for folder {index} -- {folder}")

    ### Set the required variables for this loop run
    start_time = time.time()
    updated_gaze = pd.DataFrame()
    pixel_heatmap = defaultdict(int)
    frame_no = 0
    name = folder
    folder = os.path.join(data_folder_path, folder)
    if not os.path.isdir(folder):
        continue

    csv_file = os.path.join(folder, "gaze.csv")
    gaze_df = pd.read_csv(csv_file)
    gaze_df = convert_timestamp_ns_to_ms(gaze_df)

    if not glob.glob(os.path.join(folder, "*.mp4")):
        continue
    video_file = os.path.join(folder, "*.mp4")
    video_file = glob.glob(video_file)[0]

    ref_image = cv2.imread(
        os.path.join(env_var.ROOT_PATH, env_var.ART_PIECE, "reference_image.png")
    )
    ref_image_grey = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)
    print("Existing reference image found.")

    cap = cv2.VideoCapture(video_file)
    while cap.isOpened():
        if frame_no % 100 == 0:
            print(f"Processed {frame_no} frames")

        frame_no += 1
        frame_exists, curr_frame = cap.read()

        if frame_exists:
            if is_single_color(curr_frame):
                continue

            (
                gaze_object_crop,
                closest_row,
                x_pixel,
                y_pixel,
            ) = get_closest_individual_gaze_object(
                cap, curr_frame, gaze_df, env_var.DETECT_BOUNDING_SIZE
            )

            if not gaze_object_crop.any() or closest_row.empty:
                continue

            gray_curr_frame = cv2.cvtColor(curr_frame.copy(), cv2.COLOR_BGR2GRAY)
            try:
                ref_center = reference_gaze_point_mapper(
                    gray_curr_frame, ref_image_grey, x_pixel, y_pixel
                )

                cv2.imwrite(
                    f"qc_heatmap/{name}_{frame_no}_org_points.jpg",
                    cv2.circle(
                        np.copy(curr_frame), (x_pixel, y_pixel), 15, (255, 0, 0), 15
                    ),
                )
                cv2.imwrite(
                    f"qc_heatmap/{name}_{frame_no}_new_points.jpg",
                    cv2.circle(np.copy(ref_image), ref_center, 15, (255, 0, 0), 15),
                )

            except Exception as ee:
                print(f"Error in running SIFT for frame {frame_no}")
                print(ee)
                continue

            if ref_center == None:
                continue

            closest_row["ref_center_x"] = ref_center[0]
            closest_row["ref_center_y"] = ref_center[1]
            updated_gaze = pd.concat([updated_gaze, closest_row])
            pixel_heatmap[ref_center] += 1

        else:
            break

    normalized_heatmap_dict = normalize_heatmap_dict(pixel_heatmap)
    final_img = draw_heatmap_on_ref_img(
        pixel_heatmap, np.copy(ref_image), env_var.DRAW_BOUNDING_SIZE
    )

    cap.release()

    output_path = os.path.join(output_folder_path, name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    save_outputs(
        output_path,
        name,
        ref_image,
        env_var.DETECT_BOUNDING_SIZE,
        final_img,
        updated_gaze,
    )
    end = time.time()
    print(f"Time taken for {name} is {end - start_time}")
