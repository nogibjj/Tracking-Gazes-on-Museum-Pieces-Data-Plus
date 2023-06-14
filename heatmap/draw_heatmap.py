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
from collections import defaultdict
import os
import glob
import pandas as pd
import cv2
from functions import (
    convert_timestamp_ns_to_ms,
    get_closest_individual_gaze_object,
    get_closest_reference_pixel,
    normalize_heatmap_dict,
    draw_heatmap_on_ref_img,
    create_directory,
)


# Define Constant Variables
SKIP_FIRST_N_FRAMES = 60  # As some (most) videos start with grey screen
RUN_FOR_FRAMES = 100  # Too low a value will cause a division by zero error
DETECT_BOUNDING_SIZE = 20  # Size of the bounding box for detecition
DRAW_BOUNDING_SIZE = (
    3  # Size of the bounding box (radius if circle) for the bounding box on the heatmap
)
ROOT_PATH = "/workspaces/Tracking-Gazes-on-Museum-Pieces-Data-Plus/data"
TEMP_OUTPUT_DIR = "." + os.sep + "output"
create_directory(TEMP_OUTPUT_DIR)



for index, folder in enumerate(os.listdir(ROOT_PATH)):
    folder = os.path.join(ROOT_PATH, folder)
    print(f"Running for folder {index} -- {folder}")
    pixel_heatmap = defaultdict(int)
    frame_no = 0

    # ToDo: Write a function to skip the first n frames based on percentage of grey/black color in the image
    name = folder.split(os.sep)[-1]

    csv_file = os.path.join(folder, "gaze.csv")
    video_file = os.path.join(folder, "*.mp4")
    video_file = glob.glob(video_file)[0]

    gaze_csv = pd.read_csv(csv_file)
    gaze_csv = convert_timestamp_ns_to_ms(gaze_csv)

    updated_gaze = pd.DataFrame()
    cap = cv2.VideoCapture(video_file)

    while cap.isOpened():
        if frame_no % 1000 == 0:
            print(f"Processed {frame_no} frames")

        frame_no += 1
        frame_exists, curr_frame = cap.read()

        if frame_no < SKIP_FIRST_N_FRAMES:
            continue

        ##### Uncomment below if early stopping is required
        if frame_no > SKIP_FIRST_N_FRAMES + RUN_FOR_FRAMES:
            break

        elif frame_no == SKIP_FIRST_N_FRAMES and frame_exists:
            first_frame = curr_frame

        elif frame_exists:
            gaze_object_crop, closest_row = get_closest_individual_gaze_object(
                cap, curr_frame, gaze_csv, DETECT_BOUNDING_SIZE
            )
            ref_center = get_closest_reference_pixel(first_frame, gaze_object_crop)
            closest_row["ref_center_x"] = ref_center[0]
            closest_row["ref_center_y"] = ref_center[1]
            updated_gaze = pd.concat([updated_gaze, closest_row])
            pixel_heatmap[ref_center] += 1

        else:
            break

    normalized_heatmap_dict = normalize_heatmap_dict(pixel_heatmap)
    final_img = draw_heatmap_on_ref_img(pixel_heatmap, first_frame, DRAW_BOUNDING_SIZE)

    cap.release()

    ### Write the outputs to the original data folder
    cv2.imwrite(os.path.join(ROOT_PATH, f"/reference_image_{name}.png"), first_frame)
    cv2.imwrite(
        os.path.join(ROOT_PATH, f"/heatmap_output_{name}_{DETECT_BOUNDING_SIZE}.png"),
        final_img,
    )
    updated_gaze.to_csv(
        os.path.join(ROOT_PATH, f"/updated_gaze_{name}.csv"), index=False
    )

    ### Write the data to the temp output folder
    cv2.imwrite(f"{TEMP_OUTPUT_DIR}/{name}_reference_image.png", first_frame)
    cv2.imwrite(f"{TEMP_OUTPUT_DIR}/{name}_heatmap.png", final_img)
    updated_gaze.to_csv(f"{TEMP_OUTPUT_DIR}/{name}_updated_gaze.csv", index=False)
