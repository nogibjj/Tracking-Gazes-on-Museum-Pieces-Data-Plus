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
    reference_image_finder,
    is_single_color,
)
from config.config import *

# Set env variables based on config file
try:
    env = sys.argv[1]
    # env_var = eval(env + "_config")
    env_var = eval("eric" + "_config")
except:
    print("Enter valid env variable. Refer to classes in the config.py file")
    sys.exit()

# user must create a folder structure of input_data/art_piece
data_folder_path = os.path.join(
    env_var.ROOT_PATH, env_var.ART_PIECE
)  # all the participant folders are here
output_folder_path = os.path.join(env_var.OUTPUT_PATH, env_var.ART_PIECE)

reference_frame_dict = dict()

for index, folder in enumerate(sorted(os.listdir(data_folder_path))):
    print("#" * 50)
    print(f"Extracting Reference Frame from folder {index} -- {folder}")

    ### Set the required variables for this loop run
    start_time = time.time()
    frame_no = 0
    participant_folder = os.path.join(data_folder_path, folder)

    if not (os.path.isdir(participant_folder)):
        print(f"{participant_folder} is not a folder. Skipping.")
        continue

    if not glob.glob(os.path.join(participant_folder, "*.mp4")):
        print(f"No video file found in {participant_folder}. Skipping.")
        continue

    video_file = os.path.join(participant_folder, "*.mp4")
    video_file = glob.glob(video_file)[0]

    if not (env_var.REFERENCE_IMAGE):
        print(os.path.join(env_var.ROOT_PATH, env_var.ART_PIECE, "reference_image.png"))

        ref_image = cv2.imread(
            os.path.join(env_var.ROOT_PATH, env_var.ART_PIECE, "reference_image.png")
        )
        print("Existing reference image found.")

        sys.exit()

    else:
        cap = cv2.VideoCapture(video_file)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        cap.release()

        # preserve the aspect ratio
        resize_factor = [0.5, 0.4, 0.3, 0.2, 0.1, 0.05]
        factor_found = False

        for factor in resize_factor:
            if factor_found:
                continue
            new_height = frame_height * factor

            # if this condition does not trigger
            # then the resolution of the video
            # is substantially larger than 4K
            if 200 < new_height < 500:
                new_width = frame_width * factor
                factor_found = True
                break

        ref_image, ref_image_grey = reference_image_finder(
            video_file,
            buckets=fps,
            early_stop=False,
            resize_factor=(new_width, new_height),
        )
        print("Reference image extracted from video.")

        reference_frame_dict[index] = ref_image

    # pixel_heatmap = defaultdict(int)

    # normalized_heatmap_dict = normalize_heatmap_dict(pixel_heatmap)
    # final_img = draw_heatmap_on_ref_img(
    #     pixel_heatmap, np.copy(ref_image), env_var.DRAW_BOUNDING_SIZE
    # )

    # cap.release()

    # output_path = os.path.join(output_folder_path, name)
    # if not os.path.exists(output_path):
    #     os.makedirs(output_path)

    # save_outputs(
    #     output_path,
    #     name,
    #     ref_image,
    #     env_var.DETECT_BOUNDING_SIZE,
    #     final_img,
    #     updated_gaze,
    # )
    end = time.time()
    print(f"Time taken for {folder} is {end - start_time}")
