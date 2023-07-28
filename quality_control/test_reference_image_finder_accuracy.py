import sys
import os
import time

path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# prepend parent directory to the system path:
sys.path.insert(0, path)

import glob
from heatmap.functions import (
    test_reference_image_finder,
)
from config.config import *

# Set env variables based on config file
try:
    env = sys.argv[1]
    env_var = eval(env + "_config")
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

start_time = time.time()
for index, folder in enumerate(sorted(os.listdir(data_folder_path))):
    print("#" * 50)
    print(f"Extracting Reference Frame from folder {index} -- {folder}")
    if index != 2:
        continue
    ### Set the required variables for this loop run
    participant_folder = os.path.join(data_folder_path, folder)

    if not (os.path.isdir(participant_folder)):
        print(f"{participant_folder} is not a folder. Skipping.")
        continue

    if not glob.glob(os.path.join(participant_folder, "*.mp4")):
        print(f"No video file found in {participant_folder}. Skipping.")
        continue

    video_file = os.path.join(participant_folder, "*.mp4")
    video_file = glob.glob(video_file)[0]

    check_start_time = time.time()
    test_reference_image_finder(video_file, 1)
    check_end_time = time.time()
    print(
        f"Time taken for check: {check_end_time - check_start_time} seconds for {participant_folder}"
    )


with open(
    "quality_control/ref_image_output/error_log_reference_image_creator_QC.txt", "w"
) as f:
    for line in error_log:
        f.write(str(line) + "\n")
    print("Error log written to file.")

print("Total time taken: ", time.time() - script_start_time)
print("Error log: ", error_log)
