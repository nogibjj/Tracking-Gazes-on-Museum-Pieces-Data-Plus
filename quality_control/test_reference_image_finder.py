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
from heatmap.functions import (
    reference_image_finder,
    mse,
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
reference_frame_dict_robust = dict()

script_start_time = time.time()


QC_output_path = "quality_control/ref_image_output"
error_log = []

if not os.path.exists(QC_output_path):
    raise Exception("QC_output_path does not exist")

for index, folder in enumerate(sorted(os.listdir(data_folder_path))):
    print("#" * 50)
    print(f"Extracting Reference Frame from folder {index} -- {folder}")
    if index != 2:
        continue
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
        new_height = int(frame_height * factor)

        # if this condition does not trigger
        # then the resolution of the video
        # is substantially larger than 4K
        if 200 < new_height < 500:
            new_width = int(frame_width * factor)
            factor_found = True
            break

    optimized_start_time = time.time()
    ref_image, ref_image_grey, ref_num = reference_image_finder(
        video_file,
        buckets=fps,
        early_stop=False,
        resize_factor=(new_width, new_height),
        debug=False,
    )
    optimized_end_time = time.time()
    print("Reference image extracted from video - non-robust.")
    print(
        f"Time taken for optimized: {optimized_end_time - optimized_start_time} seconds"
    )
    reference_frame_dict[index] = ref_image

    robust_start_time = time.time()
    (
        ref_image_robust,
        ref_image_grey_robust,
        ref_num_robust,
    ) = reference_image_finder(
        video_file,
        buckets=fps,
        early_stop=False,
        resize_factor=(new_width, new_height),
        debug=True,
    )
    robust_end_time = time.time()
    print("Reference image extracted from video - robust.")
    reference_frame_dict_robust[index] = ref_image_robust

    print(f"Time taken for robust: {robust_end_time - robust_start_time} seconds")
    # print(" Reference frame non-robust: ", ref_image_grey)
    # print(" Reference frame robust: ", ref_image_grey_robust)
    print("Shape of the ref image optimized : ", ref_image_grey.shape)
    print("Shape of the ref image robust : ", ref_image_grey_robust.shape)

    try:
        qc_mse = mse(ref_image_grey, ref_image_grey_robust, debug=True)
        print(f"The mse between the two images is {qc_mse}")
        assert qc_mse == 0

        if ref_num == ref_num_robust:
            print("The reference frame number is the same")

        else:
            print("The reference frame number is not the same")
            error_log.append(
                [index, "index_only, the MSE is the same. No error found."]
            )

    except AssertionError:
        print("The greyscale images are not the same")
        print("The reference frame number is not the same")
        print("The index of the non-robust image is: ", ref_num)
        print("The index of the robust image is: ", ref_num_robust)
        error_log.append([index, folder])

    cv2.imwrite(
        os.path.join(QC_output_path, f"ref_image_{index}_robust_{ref_num_robust}.png"),
        ref_image_robust,
    )
    cv2.imwrite(
        os.path.join(QC_output_path, f"ref_image_{index}_optimized_{ref_num}.png"),
        ref_image,
    )

with open(
    "quality_control/ref_image_output/error_log_reference_image_creator.txt", "w"
) as f:
    for line in error_log:
        f.write(str(line) + "\n")
    print("Error log written to file.")

print("Total time taken: ", time.time() - script_start_time)
print("Error log: ", error_log)
