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
ref_image = cv2.imread(os.path.join(env_var.ROOT_PATH, env_var.ART_PIECE, 'reference_image.png'))
width, height = ref_image.shape[0], ref_image.shape[1]

for index, folder in enumerate(os.listdir(data_folder_path)):
        folder = os.path.join(data_folder_path, folder)
        if not os.path.isdir(folder):
            continue
        print(f'Running for folder -- {folder}')
        name = folder.split(os.sep)[-1]
        updated_gaze = os.path.join(folder, "updated_gaze*.csv")
        updated_gaze = glob.glob(updated_gaze)[0]
        gaze_csv = pd.read_csv(updated_gaze)

        output_path = os.path.join(output_folder_path, name)
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        gaze_csv['timestamp [ns]'] = pd.to_datetime(gaze_csv['timestamp [ns]'])
        start_timestamp = gaze_csv['timestamp [ns]'][0]
        gaze_csv['timestamp [ns]'] = (gaze_csv['timestamp [ns]'] - start_timestamp)
        gaze_csv['timestamp [ns]'] = gaze_csv['timestamp [ns]'].astype(np.int64) / int(1e6)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        video = cv2.VideoWriter(f'{output_path}/scanpath_ref_image.mp4', fourcc, 30, (width, height))

        queue = []
        keep_last = 15

        for index, row in gaze_csv.iterrows():
                x_pixel = round(row['ref_center_x'])
                y_pixel = round(row['ref_center_y'])

                queue.append((x_pixel, y_pixel))
        
                if len(queue) > keep_last:
                    ref_image = cv2.imread(os.path.join(env_var.ROOT_PATH, env_var.ART_PIECE, 'reference_image.png'))
                    ref_image = cv2.circle(ref_image, queue[-1], 15, (0,0,255), 2)
                    for idx, point in enumerate(queue):
                        if idx !=0 :
                            ref_image = cv2.line(ref_image, point, queue[idx-1], (255, 255, 255), 3)

                    queue.pop(0) 

                    ref_image = cv2.resize(ref_image, (width, height), interpolation = cv2.INTER_AREA)
                    video.write(ref_image)     
        video.release()