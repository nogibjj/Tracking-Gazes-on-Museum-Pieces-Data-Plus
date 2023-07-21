import cv2
import pandas as pd
import numpy as np
import sys
import os
import glob

# prepend parent directory to the system path:
path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, path)

from config.config import *

try:
    env = sys.argv[1]
    env_var = eval(env + "_config")
except Exception as ee:
    print("Enter valid env variable. Refer to classes in the config.py file")
    sys.exit()

output_folder_path = os.path.join(env_var.OUTPUT_PATH, env_var.ART_PIECE)
ref_image = cv2.imread(os.path.join(env_var.ROOT_PATH, env_var.ART_PIECE, "reference_image.png"))
width, height = ref_image.shape[0], ref_image.shape[1]

for index, folder in enumerate(os.listdir(output_folder_path)):
    folder = os.path.join(output_folder_path, folder)
    if not os.path.isdir(folder):
        continue
    print(f"Running for folder -- {folder}")
    name = folder.split(os.sep)[-1]

    if not glob.glob(os.path.join(folder, "updated_gaze*.csv")):
        continue
    updated_gaze = os.path.join(folder, "updated_gaze*.csv")
    updated_gaze = glob.glob(updated_gaze)[0]

    gaze_csv = pd.read_csv(updated_gaze)

    output_path = os.path.join(output_folder_path, name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)


    gaze_csv["timestamp [ns]"] = pd.to_datetime(gaze_csv["timestamp [ns]"])
    start_timestamp = gaze_csv["timestamp [ns]"][0]
    gaze_csv["timestamp [ns]"] = gaze_csv["timestamp [ns]"] - start_timestamp
    gaze_csv["timestamp [ns]"] = gaze_csv["timestamp [ns]"].astype(np.int64) / int(1e6)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(
        f"temp_{name}.mp4", fourcc, 30, (width*2, height)
    )

    queue_orig = []
    queue_sift = []
    keep_last = 15

    for index, row in gaze_csv.iterrows():
        if index % 150 == 0:
            print(f"Running for row {index}")
        x_orig = round(row["gaze x [px]"])
        y_orig = round(row["gaze y [px]"])
        queue_orig.append((x_orig, y_orig))

        x_sift = round(row["ref_center_x"])
        y_sift = round(row["ref_center_y"])
        queue_sift.append((x_sift, y_sift))

        if len(queue_sift) > keep_last:
            orig_ref = cv2.imread(os.path.join(env_var.ROOT_PATH, env_var.ART_PIECE, "reference_image.png"))
            orig_ref = cv2.circle(orig_ref, queue_orig[-1], 15, (0, 0, 255), 2)
            for idx, point in enumerate(queue_orig):
                if idx != 0:
                    orig_ref = cv2.line(orig_ref, point, queue_orig[idx - 1], (255, 255, 255), 3)
            orig_ref = cv2.resize(orig_ref, (width, height), interpolation=cv2.INTER_AREA)
            queue_orig.pop(0)

            sift_ref = cv2.imread(os.path.join(env_var.ROOT_PATH, env_var.ART_PIECE, "reference_image.png"))
            sift_ref = cv2.circle(sift_ref, queue_sift[-1], 15, (0, 0, 255), 2)
            for idx, point in enumerate(queue_sift):
                if idx != 0:
                    sift_ref = cv2.line(sift_ref, point, queue_sift[idx - 1], (255, 255, 255), 3)
            sift_ref = cv2.resize(sift_ref, (width, height), interpolation=cv2.INTER_AREA)
            queue_sift.pop(0)
  
            final_op = cv2.hconcat([orig_ref, sift_ref])
            video.write(final_op)
    video.release()
