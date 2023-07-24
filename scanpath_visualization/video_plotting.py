import cv2
import pandas as pd
import numpy as np
import sys
import os
import glob
import traceback

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

data_folder_path = os.path.join(env_var.ROOT_PATH, env_var.ART_PIECE)
output_folder_path = os.path.join(env_var.OUTPUT_PATH, env_var.ART_PIECE)

for index, folder in enumerate(os.listdir(data_folder_path)):
    folder = os.path.join(data_folder_path, folder)
    if not os.path.isdir(folder):
        continue
    print(f"Running for folder - {folder}")

    name = folder.split(os.sep)[-1]
    csv_file = os.path.join(folder, "gaze.csv")
    video_file = os.path.join(folder, "*.mp4")
    video_file = glob.glob(video_file)[0]
    gaze_csv = pd.read_csv(csv_file)

    output_path = os.path.join(output_folder_path, name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    gaze_csv["timestamp [ns]"] = pd.to_datetime(gaze_csv["timestamp [ns]"])
    start_timestamp = gaze_csv["timestamp [ns]"][0]
    gaze_csv["timestamp [ns]"] = gaze_csv["timestamp [ns]"] - start_timestamp
    gaze_csv["timestamp [ns]"] = gaze_csv["timestamp [ns]"].astype(np.int64) / int(1e6)

    rolling_window = 30
    gaze_csv["x_smooth"] = gaze_csv["gaze x [px]"].rolling(rolling_window).mean()
    gaze_csv["y_smooth"] = gaze_csv["gaze y [px]"].rolling(rolling_window).mean()
    gaze_csv = gaze_csv[rolling_window:]
    fixation_centers = (
        gaze_csv[["x_smooth", "y_smooth", "fixation id"]]
        .groupby("fixation id")
        .mean()
        .reset_index()
    )

    cap = cv2.VideoCapture(video_file)
    frame_exists, curr_frame = cap.read()

    height, width, layers = curr_frame.shape

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(
        f"{output_path}/scanpath_video.mp4", fourcc, 30, (width, height)
    )

    frame_no = 1
    prev_point = None
    queue = []
    keep_last = 15

    while cap.isOpened():
        frame_exists, curr_frame = cap.read()
        if frame_exists:
            current_timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
            closet_value = min(
                gaze_csv["timestamp [ns]"], key=lambda x: abs(x - current_timestamp)
            )
            closest_row = gaze_csv[
                gaze_csv["timestamp [ns]"] == closet_value
            ].reset_index()

            x_pixel = round(closest_row["x_smooth"][0])
            y_pixel = round(closest_row["y_smooth"][0])
            fixation_id = closest_row["fixation id"][0]

            queue.append((x_pixel, y_pixel))

            if len(queue) > keep_last:
                curr_frame = cv2.circle(curr_frame, queue[-1], 15, (0, 0, 255), 2)
                for idx, point in enumerate(queue):
                    if idx != 0:
                        curr_frame = cv2.line(
                            curr_frame, point, queue[idx - 1], (255, 255, 255), 3
                        )

                queue.pop(0)
            video.write(curr_frame)

        else:
            break

        frame_no += 1

    cap.release()
    video.release()